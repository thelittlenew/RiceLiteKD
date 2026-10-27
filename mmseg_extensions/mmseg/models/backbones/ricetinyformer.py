# -*- coding: utf-8 -*-
"""mmseg 骨干适配器：Stage-3 RiceTinyFormer 编码器。

用途：把自研 Stage-3 学生（RiceTinyFormer）的**编码器**接进官方 mmsegmentation 的
SegFormer 配方（SegformerHead），使其与官方 SegFormer-B3（MixVisionTransformer）在
**同一框架、同一配方、同一数据划分**下做单变量（仅骨干）对照。

模块命名与形状与 `mask2former_backbone_compare.py` 中的 RiceTinyFormerBackbone
逐一对齐（stem + stage2~stage5 + proj2~proj5），因此可直接加载 Stage-3 checkpoint
编码器部分的 468 个张量；Stage-3 自带的 fpn_fuse / seg_head 不在此模块内。

输出（输入 512x512 时）：
    P2 [B,128,128,128] stride 4
    P3 [B,128, 64, 64] stride 8
    P4 [B,128, 32, 32] stride 16
    P5 [B,128, 16, 16] stride 32
"""
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class DSConvBlock(nn.Module):
    """Inverted bottleneck + depthwise conv，与 Stage-3 RiceTinyFormer 完全一致。"""

    def __init__(self, in_ch, out_ch, stride=1, expand=4):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.pw1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1,
                            groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = F.gelu(self.bn1(self.pw1(x)))
        y = F.gelu(self.bn2(self.dw(y)))
        y = self.bn3(self.pw2(y))
        if self.use_res:
            y = y + x
        return y


class LiteTransformerBlock(nn.Module):
    """轻量 Transformer block（只出现在 Stage-3 的 stage4/stage5）。"""

    def __init__(self, dim, heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        t = self.norm1(x)
        attn_out, _ = self.attn(t, t, t, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x.transpose(1, 2).reshape(b, c, h, w)


@MODELS.register_module()
class RiceTinyFormer(BaseModule):
    """Stage-3 RiceTinyFormer 编码器（mmseg backbone 接口）。

    Args:
        out_dim (int): 四个尺度投影后的通道数，Stage-3 固定 128。
        in_channels (int): 输入通道，RGB = 3。
        out_indices (Sequence[int]): 输出哪几个尺度，默认 4 个全出（P2–P5）。
        init_cfg (dict | None): 默认 None（本实验用 `load_from` 加载 Stage-3 编码器权重）。
    """

    OUT_DIM = 128
    OUT_CHANNELS = [128, 128, 128, 128]

    def __init__(self, out_dim=128, in_channels=3, out_indices=(0, 1, 2, 3), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.num_features = [out_dim] * 4

        # 512 -> 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            DSConvBlock(48, 64, expand=4),
            DSConvBlock(64, 64, expand=4),
            DSConvBlock(64, 64, expand=4),
        )
        # 256 -> 128
        self.stage2 = nn.Sequential(
            DSConvBlock(64, 96, stride=2, expand=4),
            DSConvBlock(96, 96, expand=4),
            DSConvBlock(96, 96, expand=4),
            DSConvBlock(96, 96, expand=4),
            DSConvBlock(96, 96, expand=4),
        )
        # 128 -> 64
        self.stage3 = nn.Sequential(
            DSConvBlock(96, 128, stride=2, expand=4),
            DSConvBlock(128, 128, expand=4),
            DSConvBlock(128, 128, expand=4),
            DSConvBlock(128, 128, expand=4),
            DSConvBlock(128, 128, expand=4),
        )
        # 64 -> 32
        self.stage4 = nn.Sequential(
            DSConvBlock(128, 160, stride=2, expand=4),
            DSConvBlock(160, 160, expand=4),
            DSConvBlock(160, 160, expand=4),
            DSConvBlock(160, 160, expand=4),
            LiteTransformerBlock(160, heads=5, mlp_ratio=2.0),
            LiteTransformerBlock(160, heads=5, mlp_ratio=2.0),
        )
        # 32 -> 16
        self.stage5 = nn.Sequential(
            DSConvBlock(160, 192, stride=2, expand=4),
            DSConvBlock(192, 192, expand=4),
            DSConvBlock(192, 192, expand=4),
            DSConvBlock(192, 192, expand=4),
            LiteTransformerBlock(192, heads=6, mlp_ratio=2.0),
            LiteTransformerBlock(192, heads=6, mlp_ratio=2.0),
            LiteTransformerBlock(192, heads=6, mlp_ratio=2.0),
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(96, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.GELU())
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.GELU())
        self.proj4 = nn.Sequential(
            nn.Conv2d(160, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.GELU())
        self.proj5 = nn.Sequential(
            nn.Conv2d(192, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.GELU())

    def forward(self, x):
        x = self.stem(x)
        p2_raw = self.stage2(x)
        p3_raw = self.stage3(p2_raw)
        p4_raw = self.stage4(p3_raw)
        p5_raw = self.stage5(p4_raw)
        outs = (self.proj2(p2_raw), self.proj3(p3_raw),
                self.proj4(p4_raw), self.proj5(p5_raw))
        return tuple(outs[i] for i in self.out_indices)

    def init_weights(self):
        """仅在未使用 init_cfg / load_from 时生效（本实验走 load_from）。"""
        if self.init_cfg is not None:
            super().init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for blk in (self.stage4[4], self.stage4[5], self.stage5[4],
                    self.stage5[5], self.stage5[6]):
            nn.init.xavier_uniform_(blk.attn.in_proj_weight)
            nn.init.zeros_(blk.attn.in_proj_bias)
            nn.init.xavier_uniform_(blk.attn.out_proj.weight)
            nn.init.zeros_(blk.attn.out_proj.bias)
