#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3-S/16 Stage-1 Domain Adaptive SSL for UAV Rice Field Images
===================================================================
目标：
  使用 DINOv3 ViT-S/16 预训练权重，在 15W 张 512x512 无标签航拍水稻图上做领域继续预训练，
  得到一个用于第二阶段稻穗分割的垂直领域 backbone。
推荐设备：单卡 RTX 3090 24GB
训练方式：
  - student / teacher 双网络, teacher EMA 更新
  - teacher 输入弱增强图, student 输入强增强图 + patch mask
  - CLS token: DINO self-distillation loss
  - patch tokens: masked patch feature distillation + Gram dense-structure anchoring
增强策略（按水稻生育时期自适应）：
  - vegetative (分蘖-拔节): 保守颜色增强, 较大裁剪范围
  - heading    (抽穗-灌浆): 标准增强
  - maturity   (成熟期)   : 降低 saturation/hue, 保护颜色-产量关联, 缩小裁剪范围
  - mixed      (混合时期) : 平衡参数 (默认)
新增增强：
  - RandomGamma: 模拟不同光照条件 (阴天/晴天/早晚光线)
  - RandomErasing: 模拟局部阴影/云层遮挡
保存策略：每个 epoch 都保存 checkpoint 和 backbone 导出
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from transformers import AutoModel
except Exception as e:
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = e
else:
    _TRANSFORMERS_IMPORT_ERROR = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


# ============================================================================
# Config
# ============================================================================

@dataclass
class TrainConfig:
    image_dir: str = r"/home/user2/Documents/workplace/data/Pseudo_label_training_data/Panicle_data_2025_early"
    output_dir: str = r"/home/user2/Documents/workplace/testresult"
    pretrained_path: str = r"/home/user2/Documents/workplace/"
    model_name: str = "facebook/dinov3_vits16_pretrain_lvd1689m"
    allow_online: bool = False
    image_size: int = 512
    patch_size: int = 16
    epochs: int = 40
    batch_size: int = 32
    accum_steps: int = 16
    num_workers: int = 4
    lr: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.04
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    amp: str = "fp16"
    seed: int = 42
    mask_ratio: float = 0.50
    teacher_momentum_base: float = 0.996
    teacher_temp: float = 0.04
    student_temp: float = 0.10
    center_momentum: float = 0.90
    dino_out_dim: int = 8192
    lambda_dino: float = 1.0
    lambda_patch: float = 2.0
    lambda_gram: float = 0.10
    gram_tokens: int = 256
    save_every: int = 1
    log_every: int = 50
    resume: Optional[str] = None
    compile: bool = False
    norm: str = "auto"
    growth_stage: str = "mixed"


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_norm_stats(model_name: str, norm: str):
    if norm == "imagenet":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if norm == "sat":
        return (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)
    if "sat493m" in model_name.lower():
        return (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _has_weight_file(folder: Path) -> bool:
    names = {p.name for p in folder.iterdir()} if folder.exists() and folder.is_dir() else set()
    if "model.safetensors" in names or "pytorch_model.bin" in names:
        return True
    if any(p.suffix == ".safetensors" for p in folder.glob("*.safetensors")):
        return True
    if any(p.suffix == ".bin" for p in folder.glob("*.bin")):
        return True
    return False


def _is_raw_dinov3s_pth(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        return False
    return ("dinov3_vits16" in name) and ("plus" not in name) and ("vitb16" not in name)


def find_raw_dinov3s_pth(pretrained_path: str) -> Optional[Path]:
    if pretrained_path is None or str(pretrained_path).strip() == "":
        return None
    p = Path(pretrained_path)
    if not p.exists():
        return None
    if p.is_file() and _is_raw_dinov3s_pth(p):
        return p
    if p.is_dir():
        exact = [x for x in p.iterdir() if x.is_file() and _is_raw_dinov3s_pth(x)]
        if exact:
            exact = sorted(exact, key=lambda x: ("pretrain_lvd1689m" not in x.name.lower(), x.name.lower()))
            return exact[0]
        recursive = [x for x in p.rglob("*.pth") if _is_raw_dinov3s_pth(x)]
        if recursive:
            recursive = sorted(recursive, key=lambda x: (len(x.parts), x.name.lower()))
            return recursive[0]
    return None


def write_generated_hf_config(output_dir: str) -> None:
    cfg_dir = Path(output_dir) / "generated_dinov3s_hf_config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["DINOv3ViTModel"],
        "model_type": "dinov3_vit",
        "image_size": 512, "patch_size": 16, "num_channels": 3,
        "hidden_size": 384, "intermediate_size": 1536,
        "num_hidden_layers": 12, "num_attention_heads": 6,
        "hidden_act": "gelu", "attention_dropout": 0.0,
        "initializer_range": 0.02, "layer_norm_eps": 1e-6,
        "rope_theta": 100.0, "query_bias": True, "key_bias": False,
        "value_bias": True, "proj_bias": True, "mlp_bias": True,
        "layerscale_value": 1.0, "drop_path_rate": 0.0,
        "use_gated_mlp": False, "num_register_tokens": 4,
        "pos_embed_rescale": 2.0, "apply_layernorm": True,
        "reshape_hidden_states": True, "torch_dtype": "float32",
    }
    preprocessor = {
        "do_resize": True, "size": {"height": 512, "width": 512},
        "do_rescale": True, "rescale_factor": 0.00392156862745098,
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_center_crop": False,
    }
    (cfg_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg_dir / "preprocessor_config.json").write_text(json.dumps(preprocessor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Config] 已生成配置文件: {cfg_dir}")


def resolve_pretrained_source(pretrained_path: str, model_name: str, allow_online: bool):
    if pretrained_path is not None and str(pretrained_path).strip() != "":
        p = Path(pretrained_path)
        if p.exists():
            if p.is_dir():
                if (p / "config.json").exists() and _has_weight_file(p):
                    return str(p), True, "hf_dir"
                candidates = []
                for cfg_file in p.rglob("config.json"):
                    parent = cfg_file.parent
                    if _has_weight_file(parent):
                        candidates.append(parent)
                if candidates:
                    candidates = sorted(candidates, key=lambda x: (len(x.parts), str(x)))
                    print(f"[Model] 在 pretrained_path 下自动找到 HF 模型目录: {candidates[0]}")
                    return str(candidates[0]), True, "hf_dir"
                raw_pth = find_raw_dinov3s_pth(str(p))
                if raw_pth is not None:
                    print(f"[Model] 找到原始 DINOv3-S/16 .pth 权重: {raw_pth}")
                    return str(raw_pth), True, "raw_pth"
                found = [str(x.relative_to(p)) for x in list(p.iterdir())[:30]]
                raise RuntimeError(
                    "pretrained_path 存在，但没有找到可加载的 DINOv3-S 权重。\n"
                    f"路径: {p}\n支持两种格式：\n"
                    "  A. HF 目录：config.json + model.safetensors 或 pytorch_model.bin\n"
                    "  B. 原始 .pth：dinov3_vits16_pretrain_lvd1689m-*.pth\n"
                    f"该目录前 30 个文件/子目录: {found}"
                )
            if p.is_file():
                if _is_raw_dinov3s_pth(p):
                    return str(p), True, "raw_pth"
                raise RuntimeError(
                    "你传入的是文件，但它不像 DINOv3-S/16 原始 .pth。\n"
                    f"当前文件: {p}\n"
                    "请确认文件名类似 dinov3_vits16_pretrain_lvd1689m-08c60483.pth。"
                )
    if allow_online:
        print(f"[Model] 本地 pretrained_path 不可用，允许联网，回退到: {model_name}")
        return model_name, False, "hf_online"
    raise FileNotFoundError(
        "没有找到本地 DINOv3-S 权重，且 allow_online=False。\n"
        f"pretrained_path={pretrained_path}\n"
        "请确认 D:/data_Rice/PTH 中包含 dinov3_vits16_pretrain_lvd1689m-*.pth。"
    )


def cosine_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int,
                     warmup_epochs: int = 0, start_warmup_value: float = 0.0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


def cosine_momentum_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int):
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value - (final_value - base_value) * (
        np.cos(np.pi * iters / max(len(iters), 1)) + 1
    ) / 2
    return schedule


# ============================================================================
# Dataset and augmentations (rice growth-stage aware)
# ============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_images(image_dir: str) -> List[str]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"没有在 {image_dir} 找到图像文件")
    return files


# ---------- growth-stage aware augmentation parameters ----------
_STAGE_COLOR_PARAMS = {
    "vegetative": (0.25, 0.25, 0.20, 0.03),
    "heading":    (0.40, 0.40, 0.35, 0.08),
    "maturity":   (0.30, 0.35, 0.15, 0.04),
    "mixed":      (0.35, 0.35, 0.25, 0.06),
}
_STAGE_CROP_SCALE = {
    "vegetative": (0.60, 1.00),
    "heading":    (0.70, 1.00),
    "maturity":   (0.80, 1.00),
    "mixed":      (0.70, 1.00),
}
_STAGE_GRAYSCALE_P = {
    "vegetative": 0.10,
    "heading":    0.10,
    "maturity":   0.05,
    "mixed":      0.08,
}


def _random_gamma(img: Image.Image, gamma_range=(0.7, 1.4)) -> Image.Image:
    gamma = random.uniform(*gamma_range)
    return TF.adjust_gamma(img, gamma)


class RandomGamma:
    """可序列化的 RandomGamma，替代 transforms.Lambda(lambda ...) 以兼容 Windows spawn。"""
    def __init__(self, gamma_range=(0.7, 1.4)):
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        return _random_gamma(img, self.gamma_range)


class RiceDINOv3SSLTransform:
    """
    DINO-style SSL transform with rice growth-stage aware augmentation.
    teacher_img: weak augmentation (shared geometric + mild color)
    student_img: strong augmentation (shared geometric + strong color + gamma + erasing)

    增强策略针对水稻不同生育时期自适应调整：
      - 营养生长期 (vegetative)：保守颜色增强，允许较大裁剪
      - 抽穗灌浆期 (heading)：标准增强
      - 成熟期 (maturity)：降低 sat/hue 保护颜色-产量关联，缩小裁剪范围
      - 混合时期 (mixed)：平衡参数 (默认)
    """

    def __init__(
        self,
        size: int = 512,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        growth_stage: str = "mixed",
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.growth_stage = growth_stage

        b, c, s, h = _STAGE_COLOR_PARAMS.get(growth_stage, _STAGE_COLOR_PARAMS["mixed"])
        crop_scale = _STAGE_CROP_SCALE.get(growth_stage, _STAGE_CROP_SCALE["mixed"])
        gray_p = _STAGE_GRAYSCALE_P.get(growth_stage, _STAGE_GRAYSCALE_P["mixed"])

        self.crop_scale = crop_scale

        # teacher: weak color augmentation
        self.teacher_color = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=b * 0.5, contrast=c * 0.5, saturation=s * 0.5, hue=h * 0.5)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 1.0))
            ], p=0.15),
        ])

        # student: strong color augmentation
        self.student_color = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
            ], p=0.8),
            transforms.RandomGrayscale(p=gray_p),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 1.8))
            ], p=0.35),
            # Simulate lighting variation (shadow, sun angle, cloudy vs sunny)
            transforms.RandomApply([
                RandomGamma((0.7, 1.4))
            ], p=0.25),
        ])

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # RandomErasing for simulating local shadows/cloud occlusion (applied after ToTensor)
        self.random_erasing = transforms.RandomApply([
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
        ], p=0.12)

    def _shared_geometric(self, img: Image.Image) -> Image.Image:
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=self.crop_scale, ratio=(0.90, 1.10),
        )
        img = TF.resized_crop(
            img, i, j, h, w,
            size=[self.size, self.size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.3:
            img = TF.vflip(img)
        if random.random() < 0.4:
            k = random.choice([1, 2, 3])
            img = img.rotate(90 * k, expand=False)
        return img

    def __call__(self, img: Image.Image):
        img = self._shared_geometric(img)
        teacher = self.teacher_color(img.copy())
        student = self.student_color(img.copy())
        teacher = self.to_tensor_norm(teacher)
        student = self.to_tensor_norm(student)
        # Apply RandomErasing on student tensor (simulates local shadow/occlusion)
        student = self.random_erasing(student)
        return teacher, student


class ImageFolderSSL(Dataset):
    def __init__(self, image_dir: str, transform: RiceDINOv3SSLTransform):
        self.image_paths = scan_images(image_dir)
        self.transform = transform
        print(f"[Dataset] 图像数量: {len(self.image_paths):,}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int):
        for _ in range(5):
            path = self.image_paths[index]
            try:
                img = self._read_image(path)
                teacher, student = self.transform(img)
                return teacher, student, path
            except Exception:
                index = random.randint(0, len(self.image_paths) - 1)
        raise RuntimeError("连续读取图像失败，请检查数据集是否有大量损坏图片。")


# ============================================================================
# Patch masking
# ============================================================================

def make_random_patch_mask(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device):
    num_mask = max(1, int(num_patches * mask_ratio))
    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_mask], True)
    return mask


def apply_patch_mask_to_image(x: torch.Tensor, patch_mask: torch.Tensor, patch_size: int):
    b, c, h, w = x.shape
    gh = h // patch_size
    gw = w // patch_size
    assert gh * gw == patch_mask.shape[1], "patch_mask 与图像尺寸不匹配"
    pixel_mask = patch_mask.view(b, 1, gh, gw)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return x.masked_fill(pixel_mask, 0.0)


# ============================================================================
# Raw Meta/TorchHub DINOv3-S/16 model loader
# ============================================================================

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


def _apply_rope_patch_tokens(q, k, sin, cos):
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

    def forward(self, x: torch.Tensor, rope=None):
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

    def forward(self, x: torch.Tensor, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class RawDINOv3Small16(nn.Module):
    """
    兼容 Meta/TorchHub 原始 .pth 的 DINOv3 ViT-S/16。
    关键 state_dict 名称保持为 cls_token/storage_tokens/patch_embed/blocks/norm。
    """
    def __init__(self, image_size: int = 512, patch_size: int = 16, num_register_tokens: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=384, embed_dim=384, patch_size=patch_size,
            num_register_tokens=num_register_tokens, image_size=image_size,
            num_hidden_layers=12, num_attention_heads=6,
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

    def _rope(self, h_patches: int, w_patches: int, device: torch.device, dtype):
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

    def forward(self, pixel_values=None, return_dict=True, bool_masked_pos=None, **kwargs):
        x = pixel_values
        b, _, h, w = x.shape
        patch_tokens = self.patch_embed(x)
        if bool_masked_pos is not None:
            mt = self.mask_token.to(patch_tokens.dtype).unsqueeze(0)
            patch_tokens = torch.where(bool_masked_pos.unsqueeze(-1), mt, patch_tokens)
        cls = self.cls_token.to(patch_tokens.dtype).expand(b, -1, -1)
        storage = self.storage_tokens.to(patch_tokens.dtype).expand(b, -1, -1)
        tokens = torch.cat([cls, storage, patch_tokens], dim=1)
        rope = self._rope(h // self.patch_size, w // self.patch_size, x.device, tokens.dtype)
        for blk in self.blocks:
            tokens = blk(tokens, rope=rope)
        tokens = self.norm(tokens)
        if return_dict:
            return SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens[:, 0])
        return (tokens, tokens[:, 0])

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "model_type": "raw_dinov3_vit",
            "architecture": "RawDINOv3Small16",
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": 1536,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_register_tokens": self.config.num_register_tokens,
            "layer_norm_eps": 1e-6,
            "note": "Custom raw Meta/TorchHub-compatible DINOv3-S/16. Load with RawDINOv3Small16, not AutoModel.",
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def _clean_raw_state_dict_keys(state: Dict[str, torch.Tensor]):
    cleaned = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for prefix in ["module.", "backbone.", "teacher.", "student."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        if nk.startswith("model.") and not nk.startswith("model.blocks"):
            maybe = nk[len("model."):]
            if maybe.startswith(("cls_token", "storage_tokens", "mask_token", "patch_embed", "blocks", "norm", "head")):
                nk = maybe
        cleaned[nk] = v
    return cleaned


def load_raw_dinov3s_from_pth(pth_path: str, image_size: int = 512, device=None):
    print(f"[RawPTH] 构建 DINOv3-S/16 并加载: {pth_path}")
    model = RawDINOv3Small16(image_size=image_size, patch_size=16, num_register_tokens=4)
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "teacher", "student", "backbone"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"无法解析权重文件: {pth_path}")
    state = _clean_raw_state_dict_keys(ckpt)
    model_state = model.state_dict()
    loadable = {}
    skipped_shape = []
    for k, v in state.items():
        if k in model_state:
            if tuple(v.shape) == tuple(model_state[k].shape):
                loadable[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(f"[RawPTH] loaded tensors: {len(loadable)} / model tensors: {len(model_state)}")
    print(f"[RawPTH] missing keys: {len(missing)}, unexpected ignored: {len(set(state) - set(loadable))}")
    if skipped_shape[:5]:
        print(f"[RawPTH] shape mismatch examples: {skipped_shape[:5]}")
    if len(loadable) < 120:
        raise RuntimeError(
            "原始 .pth 加载成功的张量数量过少，可能不是 DINOv3-S/16 原始权重或 key 不兼容。\n"
            f"loaded={len(loadable)}，请检查文件: {pth_path}"
        )
    if device is not None:
        model.to(device)
    return model


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256, out_dim: int = 8192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.10, teacher_temp: float = 0.04, center_momentum: float = 0.90):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_logp = F.log_softmax(student_logits / self.student_temp, dim=-1)
        teacher_p = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1).detach()
        loss = torch.sum(-teacher_p * student_logp, dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


def get_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for name in ["hidden_size", "embed_dim", "hidden_dim"]:
        v = getattr(cfg, name, None) if cfg is not None else None
        if isinstance(v, int):
            return v
    raise RuntimeError("无法从 model.config 获取 hidden_size，请检查 transformers/DINOv3 版本。")


def forward_vit_tokens(model: nn.Module, x: torch.Tensor, num_patches: int):
    outputs = model(pixel_values=x, return_dict=True)
    tokens = outputs.last_hidden_state
    cls_token = tokens[:, 0]
    patch_tokens = tokens[:, -num_patches:]
    return cls_token, patch_tokens


def masked_patch_feature_loss(student_patches: torch.Tensor, teacher_patches: torch.Tensor, patch_mask: torch.Tensor):
    s = F.normalize(student_patches, dim=-1)
    t = F.normalize(teacher_patches.detach(), dim=-1)
    loss_per_patch = 2.0 - 2.0 * (s * t).sum(dim=-1)
    mask = patch_mask.float()
    return (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)


def gram_anchor_loss(student_patches: torch.Tensor, teacher_patches: torch.Tensor, gram_tokens: int = 256):
    b, n, c = student_patches.shape
    if gram_tokens <= 0:
        return student_patches.new_tensor(0.0)
    m = min(n, gram_tokens)
    idx = torch.linspace(0, n - 1, steps=m, device=student_patches.device).long()
    s = F.normalize(student_patches[:, idx, :], dim=-1)
    t = F.normalize(teacher_patches[:, idx, :].detach(), dim=-1)
    gs = torch.bmm(s, s.transpose(1, 2))
    gt = torch.bmm(t, t.transpose(1, 2))
    return F.mse_loss(gs, gt)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


@torch.no_grad()
def copy_student_to_teacher(student: nn.Module, teacher: nn.Module) -> None:
    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)


def build_optimizer(model: nn.Module, head: nn.Module, cfg: TrainConfig):
    params = list(model.named_parameters()) + [(f"head.{n}", p) for n, p in head.named_parameters()]
    decay = []
    no_decay = []
    for name, p in params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=cfg.betas)


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    path: str, epoch: int, global_step: int,
    student: nn.Module, teacher: nn.Module,
    student_head: nn.Module, teacher_head: nn.Module,
    dino_loss: DINOLoss, optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler], cfg: TrainConfig,
) -> None:
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "student_head": student_head.state_dict(),
        "teacher_head": teacher_head.state_dict(),
        "dino_center": dino_loss.center.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    student: nn.Module, teacher: nn.Module,
    student_head: nn.Module, teacher_head: nn.Module,
    dino_loss: DINOLoss, optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler], device: torch.device,
):
    print(f"[Resume] 加载 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    student.load_state_dict(ckpt["student"], strict=True)
    teacher.load_state_dict(ckpt["teacher"], strict=True)
    student_head.load_state_dict(ckpt["student_head"], strict=True)
    teacher_head.load_state_dict(ckpt["teacher_head"], strict=True)
    if "dino_center" in ckpt:
        dino_loss.center = ckpt["dino_center"].to(device)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    print(f"[Resume] 从 epoch {start_epoch} 继续，global_step={global_step}")
    return start_epoch, global_step


def export_backbone(
    output_dir: str,
    student: nn.Module, teacher: nn.Module,
    student_head: nn.Module, teacher_head: nn.Module,
    cfg: TrainConfig,
) -> None:
    ensure_dir(output_dir)
    torch.save(
        {
            "model_name": cfg.model_name,
            "backbone": teacher.state_dict(),
            "config": asdict(cfg),
            "note": "EMA teacher backbone, recommended for stage-2 panicle segmentation.",
        },
        os.path.join(output_dir, "rice_dinov3s_teacher_backbone.pth"),
    )
    torch.save(
        {
            "model_name": cfg.model_name,
            "backbone": student.state_dict(),
            "config": asdict(cfg),
            "note": "Student backbone.",
        },
        os.path.join(output_dir, "rice_dinov3s_student_backbone.pth"),
    )
    torch.save(
        {
            "teacher_backbone": teacher.state_dict(),
            "student_backbone": student.state_dict(),
            "teacher_head": teacher_head.state_dict(),
            "student_head": student_head.state_dict(),
            "config": asdict(cfg),
        },
        os.path.join(output_dir, "rice_dinov3s_full_ssl_state.pth"),
    )
    try:
        teacher.save_pretrained(os.path.join(output_dir, "export_teacher_hf"))
        student.save_pretrained(os.path.join(output_dir, "export_student_hf"))
        print("[Export] 已导出 Hugging Face 格式目录: export_teacher_hf / export_student_hf")
    except Exception as e:
        print(f"[Export] HF 格式导出失败，但 .pth 已保存。原因: {e}")


# ============================================================================
# Training
# ============================================================================

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] 当前没有检测到 CUDA，训练会非常慢。")

    ensure_dir(cfg.output_dir)
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    save_json(asdict(cfg), os.path.join(cfg.output_dir, "args.json"))

    print("\n" + "=" * 88)
    print("DINOv3-S/16 Stage-1 Rice UAV Domain SSL")
    print("=" * 88)
    print(f"device       : {device}")
    print(f"model_name   : {cfg.model_name}")
    print(f"pretrained   : {cfg.pretrained_path}")
    print(f"allow_online : {cfg.allow_online}")
    print(f"image_dir    : {cfg.image_dir}")
    print(f"output_dir   : {cfg.output_dir}")
    print(f"image_size   : {cfg.image_size}")
    print(f"batch_size   : {cfg.batch_size}")
    print(f"accum_steps  : {cfg.accum_steps}")
    print(f"eff_batch    : {cfg.batch_size * cfg.accum_steps}")
    print(f"epochs       : {cfg.epochs}")
    print(f"lr           : {cfg.lr}")
    print(f"mask_ratio   : {cfg.mask_ratio}")
    print(f"growth_stage : {cfg.growth_stage}")
    print(f"amp          : {cfg.amp}")
    print("=" * 88 + "\n")

    pretrained_source, local_files_only, source_type = resolve_pretrained_source(
        cfg.pretrained_path, cfg.model_name, cfg.allow_online
    )
    print(f"[Model] 实际加载源: {pretrained_source}")
    print(f"[Model] source_type={source_type}, local_files_only={local_files_only}")
    write_generated_hf_config(cfg.output_dir)

    norm_key = pretrained_source if isinstance(pretrained_source, str) else cfg.model_name
    mean, std = get_norm_stats(norm_key, cfg.norm)
    print(f"[Transform] mean={mean}, std={std}")

    # Print augmentation info
    b, c, s, h = _STAGE_COLOR_PARAMS.get(cfg.growth_stage, _STAGE_COLOR_PARAMS["mixed"])
    crop_scale = _STAGE_CROP_SCALE.get(cfg.growth_stage, _STAGE_CROP_SCALE["mixed"])
    gray_p = _STAGE_GRAYSCALE_P.get(cfg.growth_stage, _STAGE_GRAYSCALE_P["mixed"])
    print(f"[Augmentation] growth_stage={cfg.growth_stage}")
    print(f"[Augmentation] ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h})")
    print(f"[Augmentation] RandomResizedCrop(scale={crop_scale})")
    print(f"[Augmentation] RandomGrayscale(p={gray_p})")
    print(f"[Augmentation] Extra: RandomGamma + RandomErasing (shadow simulation)\n")

    transform = RiceDINOv3SSLTransform(
        size=cfg.image_size, mean=mean, std=std, growth_stage=cfg.growth_stage,
    )
    dataset = ImageFolderSSL(cfg.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=8,
    )

    niter_per_epoch = len(loader)
    if niter_per_epoch == 0:
        raise RuntimeError("DataLoader 为空，请降低 batch_size 或检查数据。")

    num_patches_side = cfg.image_size // cfg.patch_size
    num_patches = num_patches_side * num_patches_side
    print(f"[Patch] grid={num_patches_side}x{num_patches_side}, num_patches={num_patches}")

    print("[Model] 加载 DINOv3 student / teacher ...")
    if source_type == "raw_pth":
        student = load_raw_dinov3s_from_pth(pretrained_source, image_size=cfg.image_size)
        teacher = load_raw_dinov3s_from_pth(pretrained_source, image_size=cfg.image_size)
    else:
        if AutoModel is None:
            print("[FATAL] transformers 导入失败，请安装：pip install -U 'transformers>=4.56.0'")
            print(f"原始错误: {_TRANSFORMERS_IMPORT_ERROR}")
            sys.exit(1)
        student = AutoModel.from_pretrained(pretrained_source, local_files_only=local_files_only)
        teacher = AutoModel.from_pretrained(pretrained_source, local_files_only=local_files_only)
    hidden_size = get_hidden_size(student)

    student_head = DINOProjectionHead(hidden_size, out_dim=cfg.dino_out_dim)
    teacher_head = DINOProjectionHead(hidden_size, out_dim=cfg.dino_out_dim)

    copy_student_to_teacher(student, teacher)
    copy_student_to_teacher(student_head, teacher_head)

    student.to(device)
    teacher.to(device)
    student_head.to(device)
    teacher_head.to(device)

    teacher.eval()
    teacher_head.eval()

    if cfg.compile and hasattr(torch, "compile"):
        print("[Model] torch.compile enabled")
        student = torch.compile(student)
        student_head = torch.compile(student_head)

    total_backbone_params = count_params(student)
    trainable_params = count_params(student) + count_params(student_head)
    print(f"[Model] hidden_size={hidden_size}")
    print(f"[Model] backbone params: {total_backbone_params / 1e6:.2f}M")
    print(f"[Model] trainable student+head params: {trainable_params / 1e6:.2f}M")

    dino_loss_fn = DINOLoss(
        out_dim=cfg.dino_out_dim,
        student_temp=cfg.student_temp,
        teacher_temp=cfg.teacher_temp,
        center_momentum=cfg.center_momentum,
    ).to(device)

    optimizer = build_optimizer(student, student_head, cfg)
    scaler = GradScaler(enabled=(cfg.amp == "fp16" and device.type == "cuda"))

    lr_schedule = cosine_scheduler(
        base_value=cfg.lr, final_value=cfg.min_lr,
        epochs=cfg.epochs, niter_per_ep=niter_per_epoch,
        warmup_epochs=cfg.warmup_epochs, start_warmup_value=cfg.lr * 0.01,
    )
    momentum_schedule = cosine_momentum_scheduler(
        base_value=cfg.teacher_momentum_base, final_value=1.0,
        epochs=cfg.epochs, niter_per_ep=niter_per_epoch,
    )

    start_epoch = 1
    global_step = 0
    if cfg.resume is not None and os.path.exists(cfg.resume):
        start_epoch, global_step = load_checkpoint(
            cfg.resume, student, teacher, student_head, teacher_head,
            dino_loss_fn, optimizer, scaler, device,
        )

    history_path = os.path.join(cfg.output_dir, "history.jsonl")

    print("\n[Train] 开始训练")
    print("-" * 120)
    header = (
        f"{'Epoch':>5} | {'Iter':>7} | {'Loss':>9} | {'DINO':>9} | {'Patch':>9} | "
        f"{'Gram':>9} | {'LR':>10} | {'Mom':>8} | {'MemGB':>7}"
    )
    print(header)
    print("-" * 120)

    for epoch in range(start_epoch, cfg.epochs + 1):
        student.train()
        student_head.train()
        teacher.eval()
        teacher_head.eval()

        epoch_loss = 0.0
        epoch_dino = 0.0
        epoch_patch = 0.0
        epoch_gram = 0.0
        count = 0

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        for it, (teacher_img, student_img, _) in enumerate(pbar):
            step_index = (epoch - 1) * niter_per_epoch + it
            if step_index < len(lr_schedule):
                lr_now = float(lr_schedule[step_index])
            else:
                lr_now = cfg.min_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            teacher_img = teacher_img.to(device, non_blocking=True)
            student_img = student_img.to(device, non_blocking=True)

            bsz = student_img.size(0)
            patch_mask = make_random_patch_mask(
                batch_size=bsz, num_patches=num_patches,
                mask_ratio=cfg.mask_ratio, device=device,
            )
            student_img_masked = apply_patch_mask_to_image(student_img, patch_mask, cfg.patch_size)

            use_amp = (cfg.amp == "fp16" and device.type == "cuda")
            with autocast(enabled=use_amp):
                with torch.no_grad():
                    t_cls, t_patch = forward_vit_tokens(teacher, teacher_img, num_patches)
                    t_logits = teacher_head(t_cls)

                s_cls, s_patch = forward_vit_tokens(student, student_img_masked, num_patches)
                s_logits = student_head(s_cls)

                loss_dino = dino_loss_fn(s_logits, t_logits)
                loss_patch = masked_patch_feature_loss(s_patch, t_patch, patch_mask)
                loss_gram = gram_anchor_loss(s_patch, t_patch, cfg.gram_tokens)

                loss = (
                    cfg.lambda_dino * loss_dino
                    + cfg.lambda_patch * loss_patch
                    + cfg.lambda_gram * loss_gram
                )
                loss_to_backward = loss / cfg.accum_steps

            scaler.scale(loss_to_backward).backward()
            dino_loss_fn.update_center(t_logits.detach())

            do_step = ((it + 1) % cfg.accum_steps == 0) or ((it + 1) == niter_per_epoch)
            if do_step:
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(student.parameters()) + list(student_head.parameters()), cfg.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                mom = float(momentum_schedule[min(step_index, len(momentum_schedule) - 1)])
                update_teacher(student, teacher, mom)
                update_teacher(student_head, teacher_head, mom)
                global_step += 1
            else:
                mom = float(momentum_schedule[min(step_index, len(momentum_schedule) - 1)])

            loss_v = float(loss.detach().cpu())
            dino_v = float(loss_dino.detach().cpu())
            patch_v = float(loss_patch.detach().cpu())
            gram_v = float(loss_gram.detach().cpu())

            epoch_loss += loss_v * bsz
            epoch_dino += dino_v * bsz
            epoch_patch += patch_v * bsz
            epoch_gram += gram_v * bsz
            count += bsz

            mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            pbar.set_postfix({
                "loss": f"{loss_v:.4f}", "dino": f"{dino_v:.4f}",
                "patch": f"{patch_v:.4f}", "lr": f"{lr_now:.2e}", "mem": f"{mem_gb:.1f}G",
            })

            if (it + 1) % cfg.log_every == 0:
                print(
                    f"{epoch:5d} | {it + 1:7d} | {loss_v:9.4f} | {dino_v:9.4f} | "
                    f"{patch_v:9.4f} | {gram_v:9.4f} | {lr_now:10.2e} | {mom:8.5f} | {mem_gb:7.2f}"
                )

        avg = {
            "epoch": epoch, "global_step": global_step,
            "loss": epoch_loss / max(count, 1),
            "dino": epoch_dino / max(count, 1),
            "patch": epoch_patch / max(count, 1),
            "gram": epoch_gram / max(count, 1),
            "lr": lr_now, "teacher_momentum": mom,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(avg, history_path)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={avg['loss']:.5f}, dino={avg['dino']:.5f}, "
            f"patch={avg['patch']:.5f}, gram={avg['gram']:.5f}, "
            f"lr={avg['lr']:.2e}, mom={avg['teacher_momentum']:.5f}"
        )

        # ---- 每个 epoch 都保存 checkpoint 和 backbone ----
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pth"),
            epoch, global_step, student, teacher,
            student_head, teacher_head, dino_loss_fn, optimizer, scaler, cfg,
        )
        save_checkpoint(
            os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
            epoch, global_step, student, teacher,
            student_head, teacher_head, dino_loss_fn, optimizer, scaler, cfg,
        )
        export_backbone(cfg.output_dir, student, teacher, student_head, teacher_head, cfg)

    print("\n[Done] 训练完成，开始最终导出 ...")
    export_backbone(cfg.output_dir, student, teacher, student_head, teacher_head, cfg)
    print(f"[Done] 输出目录: {cfg.output_dir}")
    print(f"[Done] 推荐第二阶段加载: {os.path.join(cfg.output_dir, 'export_teacher_hf')}")
    print(f"[Done] 或加载 pth: {os.path.join(cfg.output_dir, 'rice_dinov3s_teacher_backbone.pth')}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("DINOv3-S/16 Stage-1 SSL for UAV Rice Images")

    parser.add_argument("--image_dir", type=str, default=TrainConfig.image_dir)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--pretrained_path", type=str, default=TrainConfig.pretrained_path)
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--allow_online", action="store_true")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accum_steps", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=20)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--amp", type=str, default="fp16", choices=["fp16", "none"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mask_ratio", type=float, default=0.50)
    parser.add_argument("--teacher_momentum_base", type=float, default=0.996)
    parser.add_argument("--teacher_temp", type=float, default=0.04)
    parser.add_argument("--student_temp", type=float, default=0.10)
    parser.add_argument("--center_momentum", type=float, default=0.90)
    parser.add_argument("--dino_out_dim", type=int, default=8192)

    parser.add_argument("--lambda_dino", type=float, default=1.0)
    parser.add_argument("--lambda_patch", type=float, default=2.0)
    parser.add_argument("--lambda_gram", type=float, default=0.10)
    parser.add_argument("--gram_tokens", type=int, default=256)

    parser.add_argument("--save_every", type=int, default=1, help="已废弃，每个 epoch 都会保存")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--norm", type=str, default="auto", choices=["auto", "imagenet", "sat"])

    parser.add_argument(
        "--growth_stage", type=str, default="mixed",
        choices=["vegetative", "heading", "maturity", "mixed"],
        help=(
            "水稻生育时期，决定增强策略: "
            "vegetative=分蘖拔节期(保守颜色增强), "
            "heading=抽穗灌浆期(标准增强), "
            "maturity=成熟期(降低sat/hue保护颜色-产量关联), "
            "mixed=混合时期(默认平衡参数)"
        ),
    )

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
