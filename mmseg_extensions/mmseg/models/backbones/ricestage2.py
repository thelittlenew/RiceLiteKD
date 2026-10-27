# -*- coding: utf-8 -*-
"""mmseg 骨干适配器：Stage-2 编码器（DINOv3-S/16 + ViT-Adapter-lite）。

用途：把 Stage-2 伪标签分割模型（`stage2_train_dinov3_mask2former_pseudolabel.py` 中的
`RawDINOv3Small16` + `ViTAdapterLite`）作为**编码器**接进官方 mmsegmentation 的 SegFormer
配方（SegformerHead），使其与官方 SegFormer-B2/B3（MixVisionTransformer）在**同一框架、
同一配方、同一数据划分**下做单变量（仅骨干）对照。

与 Stage-3 那个适配器（`ricetinyformer.py`）的关系：
  * **对外契约完全相同**：输入 512x512，输出 P2/P3/P4/P5 全 128 通道 @ stride 4/8/16/32，
    因此 `decode_head.in_channels` 同样是 [128,128,128,128]，同一份 SegFormer 配方直接可换。
  * 来源不同：本模块来自 Stage-2（DINOv3-S/16 + ViT-Adapter-lite）的编码器部分。

输出（输入 512x512 时）：
    P2 [B,128,128,128] stride 4
    P3 [B,128, 64, 64] stride 8
    P4 [B,128, 32, 32] stride 16
    P5 [B,128, 16, 16] stride 32

权重键映射（`_s2mm_prep.py` 生成 `ricestage2_seg_init.pth`）：
    Stage-2 ckpt `backbone.<x>` -> mmseg `backbone.vit.<x>`
    Stage-2 ckpt `adapter.<x>`  -> mmseg `backbone.adapter.<x>`
    Stage-2 ckpt `decoder.*`    丢弃（mmseg 用官方 SegformerHead，随机初始化）

下面 Raw* / ConvGNAct / SpatialPriorModule / ViTAdapterLite 的**代码行与 Stage-2 训练脚本逐字一致**
（只改写注释），以保证权重按名字直接迁移、且前向数值完全一致。
"""
import math
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS


# ---------------------------------------------------------------------------
# DINOv3 ViT-S/16（raw pth 兼容）——与 Stage-2 训练脚本逐字一致
# ---------------------------------------------------------------------------
class RawLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class RawPatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class RawMlp(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 1536, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_patch_tokens(q: torch.Tensor, k: torch.Tensor,
                             sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_patches = sin.shape[-2]
    prefix = q.shape[-2] - num_patches
    q_prefix, q_patch = q[:, :, :prefix, :], q[:, :, prefix:, :]
    k_prefix, k_patch = k[:, :, :prefix, :], k[:, :, prefix:, :]
    q_patch = (q_patch * cos) + (_rope_rotate_half(q_patch) * sin)
    k_patch = (k_patch * cos) + (_rope_rotate_half(k_patch) * sin)
    return torch.cat([q_prefix, q_patch], dim=-2), torch.cat([k_prefix, k_patch], dim=-2)


class RawSelfAttention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if rope is not None:
            sin, cos = rope
            q_dtype, k_dtype = q.dtype, k.dtype
            q, k = _apply_rope_patch_tokens(q.float(), k.float(), sin.float(), cos.float())
            q, k = q.to(q_dtype), k.to(k_dtype)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class RawBlock(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0, layer_scale: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RawSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=True, proj_bias=True)
        self.ls1 = RawLayerScale(dim, init_values=layer_scale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = RawMlp(dim=dim, hidden_dim=int(dim * mlp_ratio), bias=True)
        self.ls2 = RawLayerScale(dim, init_values=layer_scale)

    def forward(self, x: torch.Tensor, rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class RawDINOv3Small16(nn.Module):
    """DINOv3 ViT-S/16, raw pth compatible."""

    def __init__(self, image_size: int = 512, patch_size: int = 16, num_register_tokens: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=384,
            embed_dim=384,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            image_size=image_size,
            num_hidden_layers=12,
            num_attention_heads=6,
        )
        self.num_features = 384
        self.embed_dim = 384
        self.patch_size = patch_size
        self.n_storage_tokens = num_register_tokens
        self.patch_embed = RawPatchEmbed(3, 384, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.blocks = nn.ModuleList([RawBlock(384, 6, 4.0, 1.0) for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _rope(self, h_patches: int, w_patches: int, device: torch.device,
              dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        y = torch.arange(0.5, h_patches, device=device, dtype=torch.float32) / h_patches
        x = torch.arange(0.5, w_patches, device=device, dtype=torch.float32) / w_patches
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        coords = coords * 2.0 - 1.0
        inv_freq = 1.0 / (100.0 ** torch.arange(0, 1, 4 / head_dim, device=device, dtype=torch.float32))
        angles = 2.0 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)
        sin = torch.sin(angles)[None, None, :, :].to(dtype=dtype)
        cos = torch.cos(angles)[None, None, :, :].to(dtype=dtype)
        return sin, cos

    def forward_tokens(self, pixel_values: torch.Tensor,
                       return_layers: Sequence[int] = (2, 5, 8, 11)) -> List[torch.Tensor]:
        x = pixel_values
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        patch_tokens = self.patch_embed(x)
        cls = self.cls_token.to(patch_tokens.dtype).expand(b, -1, -1)
        storage = self.storage_tokens.to(patch_tokens.dtype).expand(b, -1, -1)
        tokens = torch.cat([cls, storage, patch_tokens], dim=1)
        rope = self._rope(gh, gw, x.device, tokens.dtype)
        outs = []
        return_layers = set(return_layers)
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, rope=rope)
            if i in return_layers:
                # 对中间层也用最终 LN 做尺度对齐（与 Stage-2 脚本一致）
                t = self.norm(tokens)
                p = t[:, -gh * gw:, :]
                fmap = p.transpose(1, 2).reshape(b, self.embed_dim, gh, gw).contiguous()
                outs.append(fmap)
        return outs

    def forward(self, pixel_values: torch.Tensor = None, return_dict: bool = True, **kwargs):
        feats = self.forward_tokens(pixel_values, return_layers=(11,))
        fmap = feats[-1]
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1, keepdim=True)
        all_tokens = torch.cat([cls, tokens], dim=1)
        if return_dict:
            return SimpleNamespace(last_hidden_state=all_tokens, pooler_output=cls[:, 0])
        return (all_tokens, cls[:, 0])


# ---------------------------------------------------------------------------
# ViT-Adapter-lite + spatial prior ——与 Stage-2 训练脚本逐字一致
# ---------------------------------------------------------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1,
                 groups: int = 8, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class SpatialPriorModule(nn.Module):
    """轻量 CNN spatial prior，生成 P2/P3/P4/P5（512 输入下 128/64/32/16）。"""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(3, 64, k=7, stride=2),      # 512 -> 256
            ConvGNAct(64, 64, k=3, stride=1),
        )
        self.p2 = nn.Sequential(ConvGNAct(64, out_dim, k=3, stride=2),
                                ConvGNAct(out_dim, out_dim, k=3))          # 128
        self.p3 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2),
                                ConvGNAct(out_dim, out_dim, k=3))          # 64
        self.p4 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2),
                                ConvGNAct(out_dim, out_dim, k=3))          # 32
        self.p5 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2),
                                ConvGNAct(out_dim, out_dim, k=3))          # 16

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class ViTAdapterLite(nn.Module):
    """可单卡运行的 ViT-Adapter-lite：spatial prior + 多层 patch token 注入 + FPN 自顶向下融合。

    它不是官方 ViT-Adapter 的完整 InteractionBlock + MSDeformAttn 实现，但保留了对 plain ViT
    很关键的两点：(1) CNN 产生多尺度局部先验；(2) DINOv3 多层 patch tokens 注入 P2/P3/P4/P5。
    """

    def __init__(self, vit_dim: int = 384, adapter_dim: int = 128, dropout: float = 0.05):
        super().__init__()
        self.spm = SpatialPriorModule(out_dim=adapter_dim)
        self.vit_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_dim, adapter_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, adapter_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])
        # 学习每一层 ViT feature 注入 CNN prior 的强度，初始较小更稳
        self.gamma = nn.Parameter(torch.ones(4) * 0.5)

        self.lateral = nn.ModuleList([nn.Conv2d(adapter_dim, adapter_dim, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
        ])

    def forward(self, image: torch.Tensor, vit_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # spatial prior: [P2,P3,P4,P5]
        priors = self.spm(image)
        out = []
        for i, (prior, vf) in enumerate(zip(priors, vit_feats)):
            v = self.vit_proj[i](vf)
            if v.shape[-2:] != prior.shape[-2:]:
                v = F.interpolate(v, size=prior.shape[-2:], mode="bilinear", align_corners=False)
            out.append(prior + self.gamma[i].to(dtype=prior.dtype) * v)

        # FPN top-down fusion: P5 -> P2
        p2, p3, p4, p5 = [self.lateral[i](out[i]) for i in range(4)]
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2, p3, p4, p5 = self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4), self.smooth[3](p5)
        return [p2, p3, p4, p5]


@MODELS.register_module()
class RiceStage2(BaseModule):
    """Stage-2 编码器（DINOv3-S/16 + ViT-Adapter-lite）的 mmseg backbone 接口。

    Args:
        out_dim (int): 适配器输出通道，Stage-2 固定 128（对应 SegformerHead 的
            in_channels=[128,128,128,128]）。
        in_channels (int): 输入通道；Stage-2 的 patch_embed 写死 3（RGB）。
        image_size (int): 训练 crop 尺寸（仅作记录，RoPE 按实际 token 网格动态生成）。
        patch_size (int): ViT patch 尺寸，16。
        num_register_tokens (int): DINOv3 register tokens 数，4。
        return_layers (Sequence[int]): 取哪 4 个 ViT block 的 patch token 注入金字塔，
            Stage-2 用 (2,5,8,11)。
        out_indices (Sequence[int]): 输出哪几个尺度，默认 P2–P5 全出。
        init_cfg (dict | None): 默认 None（本实验用 `load_from` 加载 ricestage2_seg_init.pth）。
    """

    OUT_CHANNELS = [128, 128, 128, 128]
    OUT_STRIDES = [4, 8, 16, 32]

    def __init__(self, out_dim=128, in_channels=3, image_size=512, patch_size=16,
                 num_register_tokens=4, return_layers=(2, 5, 8, 11),
                 out_indices=(0, 1, 2, 3), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert in_channels == 3, 'Stage-2 patch_embed 固定 3 通道'
        assert out_dim == 128, 'Stage-2 适配器固定 128 通道'
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.return_layers = tuple(return_layers)
        self.out_indices = out_indices
        self.num_features = [out_dim] * 4

        # 与 Stage-2 训练脚本同名的两个子模块；权重键由 _s2mm_prep.py 从 Stage-2 ckpt 的
        # backbone.* / adapter.* 映射到 mmseg 的 backbone.vit.* / backbone.adapter.*
        self.vit = RawDINOv3Small16(image_size=image_size, patch_size=patch_size,
                                    num_register_tokens=num_register_tokens)
        self.adapter = ViTAdapterLite(vit_dim=self.vit.embed_dim, adapter_dim=out_dim,
                                      dropout=0.05)

    def forward(self, x):
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0, \
            'H/W 必须是 patch_size 的整数倍，当前 %s' % (tuple(x.shape[-2:]),)
        vit_feats = self.vit.forward_tokens(x, return_layers=self.return_layers)
        outs = self.adapter(x, vit_feats)
        return tuple(outs[i] for i in self.out_indices)

    def init_weights(self):
        """仅在未使用 init_cfg / load_from 时生效（本实验走 load_from）。

        Stage-2 脚本里 ViT 走 `RawDINOv3Small16._init_weights`，适配器/SPM 用 PyTorch 默认
        初始化（未覆写），这里保持一致。
        """
        if self.init_cfg is not None:
            super().init_weights()
            return
        self.vit._init_weights()

