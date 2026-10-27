#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stage-2 rice panicle segmentation: DINOv3-S/16 OFFICIAL .pth -> Mask2Former-style head
======================================================================================

用途：
  1. 对比实验专用：跳过 Stage-1 领域预训练，直接用 Meta 官方
     dinov3_vits16_pretrain_lvd1689m-08c60483.pth 原始权重作为 backbone；
  2. 其余结构（ViT-Adapter-lite + Mask2Former-style decoder）与完整 Stage-2 完全一致；
  3. 数据加载、增强、loss、评估指标、resume 逻辑与 stage2 脚本保持相同，
     确保唯一变量是"是否经过 Stage-1 领域 SSL"。

推荐运行：
  python stage2_dinov3_mask2former_official_pth_train.py ^
    --backbone_ckpt D:\data_Rice\PTH\dinov3_vits16_pretrain_lvd1689m-08c60483.pth ^
    --image_dir D:\data_Rice\no_labling_all\Panicle_data_2025_early ^
    --mask_dir D:\data_Rice\no_labling_all\mask ^
    --output_dir D:\data_Rice\RicePanicle_OfficialPTH_DINOv3S_Mask2Former ^
    --epochs 80 ^
    --batch_size 4 ^
    --amp fp16
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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
except ImportError:
    A = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


# =============================================================================
# Config
# =============================================================================

CLASS_NAMES = ["Background", "Panicle"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class TrainConfig:
    train_rgb_dir: str = r"/home/user2/Documents/workplace/data/Pseudo_label_training_data/Panicle_data_2025_early"
    train_label_dir: str = r"/home/user2/Documents/workplace/data/Pseudo_label_training_data/mask"
    test_rgb_dir: str = r"/home/user2/Documents/workplace/data/Pseudo_label_training_data/Panicle_data_2025_early"
    test_label_dir: str = r"/home/user2/Documents/workplace/data/Pseudo_label_training_data/mask"

    # 核心区别：直接加载官方原始 .pth，而非 Stage-1 导出文件
    backbone_ckpt: str = r"/home/user2/Documents/workplace/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    output_dir: str = r"/home/user2/Documents/workplace/OfficialPTH_instage2_Mask2Former"

    image_size: int = 512
    patch_size: int = 16
    num_classes: int = 2
    ignore_index: int = 255

    epochs: int = 120
    batch_size: int = 12
    num_workers: int = 12

    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    min_lr_ratio: float = 0.01
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 3
    freeze_backbone_epochs: int = 3

    amp: str = "fp16"
    seed: int = 42

    ce_weight: float = 0.0
    dice_weight: float = 2.0
    use_class_weights: bool = True
    class_weight_max: float = 5.0

    hflip_prob: float = 0.5
    vflip_prob: float = 0.3
    rotate90_prob: float = 0.3
    color_jitter_prob: float = 0.5
    blur_prob: float = 0.10
    use_albumentations: bool = True

    save_every: int = 10
    val_ratio: float = 0.05

    resume: Optional[str] = "auto"
    continue_epochs: int = 40

    skip_bad_files: bool = True
    bad_file_log: str = "bad_files.txt"


# =============================================================================
# Utilities
# =============================================================================

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


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def cosine_lr(base_lr: float, min_lr: float, epoch: int, max_epoch: int, warmup_epochs: int = 0) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epoch - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Dataset (与 stage2 完全一致)
# =============================================================================

def read_image_pil_or_cv2(path: str, mode: str = "rgb") -> np.ndarray:
    try:
        if mode == "rgb":
            return np.array(Image.open(path).convert("RGB"))
        img = Image.open(path)
        return np.array(img)
    except Exception as pil_e:
        try:
            if mode == "rgb":
                arr = cv2.imread(path, cv2.IMREAD_COLOR)
                if arr is None:
                    raise RuntimeError("cv2.imread returned None")
                return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError("cv2.imread returned None")
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            return arr
        except Exception as cv_e:
            raise RuntimeError(
                f"无法识别或读取文件: {path}\n"
                f"PIL error: {repr(pil_e)}\n"
                f"cv2 error: {repr(cv_e)}"
            )


def normalize_label_array(arr: np.ndarray, num_classes: int, ignore_index: int, path: str = "") -> np.ndarray:
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        if rgb.shape[2] >= 3 and np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 0] == rgb[..., 2]):
            return normalize_label_array(rgb[..., 0], num_classes, ignore_index, path)
        lab = (np.max(rgb, axis=-1) > 0).astype(np.int64)
        return lab

    if arr.ndim != 2:
        raise RuntimeError(f"label 维度异常，期望 HxW 或 HxWx3/4，实际 shape={arr.shape}，文件: {path}")

    arr = arr.astype(np.float32)
    uniq = np.unique(arr)

    if uniq.size == 1 and float(uniq[0]) == 0.0:
        return np.zeros_like(arr, dtype=np.int64)

    if np.all(np.isin(uniq, [0, 1])):
        return (arr > 0).astype(np.int64)

    if np.all(np.isin(uniq, [0, 255])):
        return (arr > 0).astype(np.int64)

    lab = (arr > 0).astype(np.int64)
    return lab


def validate_and_filter_pairs(
    pairs: List[Tuple[str, str]],
    cfg: TrainConfig,
    split_name: str,
) -> List[Tuple[str, str]]:
    if not cfg.skip_bad_files:
        return pairs
    good = []
    bad_lines = []
    print(f"[Dataset] 检查 {split_name} image/label 可读性，坏文件会自动跳过 ...")
    for img_path, lab_path in tqdm(pairs, desc=f"Validate-{split_name}", leave=False):
        try:
            _ = read_image_pil_or_cv2(img_path, mode="rgb")
            lab = read_image_pil_or_cv2(lab_path, mode="label")
            _ = normalize_label_array(lab, cfg.num_classes, cfg.ignore_index, lab_path)
            good.append((img_path, lab_path))
        except Exception as e:
            bad_lines.append(f"[{split_name}]\tIMG={img_path}\tLABEL={lab_path}\tERROR={str(e).replace(chr(10), ' | ')}")
    if bad_lines:
        ensure_dir(cfg.output_dir)
        log_path = os.path.join(cfg.output_dir, cfg.bad_file_log)
        with open(log_path, "a", encoding="utf-8") as f:
            for line in bad_lines:
                f.write(line + "\n")
        print(f"[Dataset] {split_name} 跳过坏样本: {len(bad_lines)}，日志: {log_path}")
    print(f"[Dataset] {split_name} 可用样本: {len(good):,} / {len(pairs):,}")
    if len(good) == 0:
        raise RuntimeError(f"{split_name} 没有可用样本，请检查数据格式。")
    return good


def build_stem_map(directory: str, exts: set) -> Dict[str, str]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    d = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            d[p.stem] = str(p)
    return d


def _mask_to_image_stem_candidates(mask_stem: str) -> List[str]:
    candidates = [mask_stem]
    s = mask_stem
    for ext in IMG_EXTS:
        if s.lower().endswith(ext):
            candidates.append(s[: -len(ext)])
    suffixes = ["_mask", "-mask", "_label", "-label", "_pseudo", "-pseudo", "_pred", "-pred", "_panicle", "-panicle"]
    for base in list(candidates):
        low = base.lower()
        for suf in suffixes:
            if low.endswith(suf):
                candidates.append(base[: -len(suf)])
    out = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_pairs(rgb_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    rgb_map_raw = build_stem_map(rgb_dir, IMG_EXTS)
    lab_map = build_stem_map(label_dir, LABEL_EXTS)
    rgb_map = {k.lower(): v for k, v in rgb_map_raw.items()}

    pairs = []
    missing_img = 0
    used_img_stems = set()
    for mask_stem, lab_path in sorted(lab_map.items()):
        img_path = None
        matched_key = None
        for cand in _mask_to_image_stem_candidates(mask_stem):
            key = cand.lower()
            if key in rgb_map:
                img_path = rgb_map[key]
                matched_key = key
                break
        if img_path is None:
            missing_img += 1
            continue
        pairs.append((img_path, lab_path))
        used_img_stems.add(matched_key)

    extra_rgb = len(set(rgb_map.keys()) - used_img_stems)
    print(f"[Dataset] mask-first pairing: masks={len(lab_map):,}, images={len(rgb_map):,}, pairs={len(pairs):,}")
    if missing_img > 0:
        print(f"[Dataset] 有 {missing_img:,} 个 mask 没找到同名原图，已跳过。")
    if extra_rgb > 0:
        print(f"[Dataset] 有 {extra_rgb:,} 张原图没有对应 mask，不参与训练。")
    if len(pairs) == 0:
        raise RuntimeError(f"没有找到匹配的 image/mask 对，请检查是否同名:\nRGB: {rgb_dir}\nMask: {label_dir}")
    return pairs


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float, seed: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if val_ratio is None or val_ratio <= 0 or len(pairs) < 2:
        return pairs, pairs
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    return train_pairs, val_pairs


def build_albumentations_transform(cfg: TrainConfig):
    if A is None:
        raise ImportError(
            "当前脚本已切换为 Albumentations 数据增强，但环境中没有安装 albumentations。\n"
            "请先安装：pip install albumentations\n"
            "如果你使用 conda 环境，请先激活 sam2-env 后再安装。"
        )

    transforms_list = [
        A.HorizontalFlip(p=cfg.hflip_prob),
        A.VerticalFlip(p=cfg.vflip_prob),
        A.RandomRotate90(p=cfg.rotate90_prob),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=1.0),
        ], p=cfg.color_jitter_prob),
        A.GaussianBlur(blur_limit=(3, 3), p=cfg.blur_prob),
    ]
    return A.Compose(transforms_list)


class RiceSEGDataset(Dataset):
    def __init__(
        self,
        rgb_dir: str,
        label_dir: str,
        cfg: TrainConfig,
        train: bool = True,
        pairs: Optional[List[Tuple[str, str]]] = None,
        already_validated: bool = False,
    ):
        self.cfg = cfg
        self.train = train
        if pairs is None:
            self.pairs = build_pairs(rgb_dir, label_dir)
        else:
            self.pairs = list(pairs)
        if not already_validated:
            self.pairs = validate_and_filter_pairs(self.pairs, cfg, "train" if train else "pseudo-val")
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.albu_transform = build_albumentations_transform(cfg) if (train and cfg.use_albumentations) else None
        if train and cfg.use_albumentations:
            print(
                f"[Augment] Albumentations enabled: hflip={cfg.hflip_prob}, vflip={cfg.vflip_prob}, "
                f"rot90={cfg.rotate90_prob}, color={cfg.color_jitter_prob}, blur={cfg.blur_prob}"
            )
        print(f"[Dataset] {'train' if train else 'test'} samples: {len(self.pairs):,}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_rgb(self, path: str) -> np.ndarray:
        arr = read_image_pil_or_cv2(path, mode="rgb")
        if arr.shape[:2] != (self.cfg.image_size, self.cfg.image_size):
            arr = cv2.resize(arr, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_LINEAR)
        return arr

    def _read_label(self, path: str) -> np.ndarray:
        arr = read_image_pil_or_cv2(path, mode="label")
        arr = normalize_label_array(arr, self.cfg.num_classes, self.cfg.ignore_index, path)
        if arr.shape[:2] != (self.cfg.image_size, self.cfg.image_size):
            arr = cv2.resize(arr, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_NEAREST)
        return arr

    def _augment(self, img: np.ndarray, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.albu_transform is None:
            return img, lab
        aug = self.albu_transform(image=img, mask=lab)
        img = np.ascontiguousarray(aug["image"])
        lab = np.ascontiguousarray(aug["mask"]).astype(np.int64)
        return img, lab

    def __getitem__(self, idx: int):
        last_err = None
        for _ in range(5):
            img_path, lab_path = self.pairs[idx]
            try:
                img = self._read_rgb(img_path)
                lab = self._read_label(lab_path)
                if self.train:
                    img, lab = self._augment(img, lab)

                img = img.astype(np.float32) / 255.0
                img = (img - self.mean) / self.std
                img_t = torch.from_numpy(img).permute(2, 0, 1).float()
                lab_t = torch.from_numpy(lab).long()
                return img_t, lab_t, os.path.basename(img_path)
            except Exception as e:
                last_err = e
                idx = random.randint(0, len(self.pairs) - 1)
        raise RuntimeError(f"连续读取样本失败，最后错误: {last_err}")


@torch.no_grad()
def compute_class_weights(
    label_dir: str,
    num_classes: int,
    ignore_index: int,
    max_weight: float = 5.0,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> torch.Tensor:
    print("[ClassWeights] 扫描训练 label 统计类别频率 ...")
    counts = np.zeros(num_classes, dtype=np.float64)
    if pairs is not None:
        files = [Path(lab_path) for _, lab_path in pairs]
    else:
        files = []
        for p in Path(label_dir).rglob("*"):
            if p.is_file() and p.suffix.lower() in LABEL_EXTS:
                files.append(p)
    bad = []
    for p in tqdm(files, desc="ClassWeights", leave=False):
        try:
            arr = read_image_pil_or_cv2(str(p), mode="label")
            arr = normalize_label_array(arr, num_classes, ignore_index, str(p))
        except Exception as e:
            bad.append((str(p), str(e).replace("\n", " | ")))
            continue
        mask = (arr != ignore_index) & (arr >= 0) & (arr < num_classes)
        if mask.any():
            counts += np.bincount(arr[mask].reshape(-1), minlength=num_classes)[:num_classes]
    if bad:
        print(f"[ClassWeights] 跳过无法读取/非索引格式 label: {len(bad)}")
        for bp, err in bad[:10]:
            print(f"  - {bp}: {err}")
        if len(bad) > 10:
            print(f"  ... 其余 {len(bad)-10} 个略")
    freq = counts / max(counts.sum(), 1.0)
    nonzero = freq[freq > 0]
    if len(nonzero) == 0:
        return torch.ones(num_classes)
    median = np.median(nonzero)
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if freq[c] > 0:
            weights[c] = float(median / freq[c])
        else:
            weights[c] = 0.0
    weights = np.clip(weights, 0.2, max_weight)
    pos = weights > 0
    weights[pos] = weights[pos] / weights[pos].mean()
    print("[ClassWeights] counts:", counts.astype(np.int64).tolist())
    print("[ClassWeights] weights:", [round(float(x), 4) for x in weights.tolist()])
    if num_classes == 2 and counts[1] <= 0:
        raise RuntimeError(
            "所有训练 mask 转换后前景像素数量为 0。\n"
            "请检查 mask 是否真的是黑底白前景，或是否是 palette PNG/反色 mask。\n"
            "建议先运行脚本自带的 --check_masks_only 查看 unique 值和前景比例。"
        )
    return torch.from_numpy(weights.astype(np.float32))


# =============================================================================
# Raw DINOv3-S/16 backbone (与 stage2 完全一致，兼容官方 .pth key)
# =============================================================================

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


def _apply_rope_patch_tokens(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _rope(self, h_patches: int, w_patches: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward_tokens(self, pixel_values: torch.Tensor, return_layers: Sequence[int] = (2, 5, 8, 11)) -> List[torch.Tensor]:
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


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for prefix in ["module.", "backbone.", "teacher.", "student.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_dinov3s_backbone(ckpt_path: str, image_size: int = 512, device: Optional[torch.device] = None) -> RawDINOv3Small16:
    """
    加载 DINOv3-S/16 backbone。

    兼容两种输入：
      A. Stage-1 导出的格式: {backbone: state_dict, ...}
      B. Meta / TorchHub 官方原始 .pth: 直接是 state_dict，
         key 形如 model.cls_token / model.blocks.0.norm1.weight 等。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"backbone_ckpt 不存在: {ckpt_path}")
    print(f"[Backbone] 构建 DINOv3-S/16 并加载: {ckpt_path}")
    model = RawDINOv3Small16(image_size=image_size, patch_size=16, num_register_tokens=4)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        # 优先尝试提取嵌套的 state_dict（Stage-1 导出格式）
        for key in ["backbone", "teacher_backbone", "model", "state_dict", "teacher", "student"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                print(f"[Backbone] 从 checkpoint['{key}'] 提取 state_dict")
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"无法解析权重文件: {ckpt_path}")

    state = _clean_state_dict_keys(ckpt)
    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
            loadable[k] = v
        elif k in model_state:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(f"[Backbone] loaded tensors: {len(loadable)} / {len(model_state)}")
    print(f"[Backbone] missing keys: {len(missing)}, ignored keys: {len(set(state) - set(loadable))}")
    if skipped[:5]:
        print(f"[Backbone] shape mismatch examples: {skipped[:5]}")

    # 官方 .pth 通常 160+ 张量；Stage-1 backbone 也类似。
    if len(loadable) < 120:
        raise RuntimeError(
            f"加载成功张量太少 loaded={len(loadable)}，可能不是 DINOv3-S/16 或 ckpt key 不兼容。"
        )
    if device is not None:
        model.to(device)
    return model


# =============================================================================
# Segmentation model (与 stage2 完全一致)
# =============================================================================

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        y_embed = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h - 1 + eps) * self.scale
            x_embed = x_embed / (w - 1 + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos.to(dtype=x.dtype)


class SpatialPriorModule(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(3, 64, k=7, stride=2),
            ConvGNAct(64, 64, k=3, stride=1),
        )
        self.p2 = nn.Sequential(ConvGNAct(64, out_dim, k=3, stride=2), ConvGNAct(out_dim, out_dim, k=3))
        self.p3 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2), ConvGNAct(out_dim, out_dim, k=3))
        self.p4 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2), ConvGNAct(out_dim, out_dim, k=3))
        self.p5 = nn.Sequential(ConvGNAct(out_dim, out_dim, k=3, stride=2), ConvGNAct(out_dim, out_dim, k=3))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class ViTAdapterLite(nn.Module):
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
        self.gamma = nn.Parameter(torch.ones(4) * 0.5)
        self.lateral = nn.ModuleList([nn.Conv2d(adapter_dim, adapter_dim, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
            ConvGNAct(adapter_dim, adapter_dim, k=3, dropout=dropout),
        ])

    def forward(self, image: torch.Tensor, vit_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        priors = self.spm(image)
        out = []
        for i, (prior, vf) in enumerate(zip(priors, vit_feats)):
            v = self.vit_proj[i](vf)
            if v.shape[-2:] != prior.shape[-2:]:
                v = F.interpolate(v, size=prior.shape[-2:], mode="bilinear", align_corners=False)
            out.append(prior + self.gamma[i].to(dtype=prior.dtype) * v)

        p2, p3, p4, p5 = [self.lateral[i](out[i]) for i in range(4)]
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2, p3, p4, p5 = self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4), self.smooth[3](p5)
        return [p2, p3, p4, p5]


class MultiScaleMask2FormerStyleHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        pixel_dim: int = 128,
        num_classes: int = 2,
        num_queries: int = 100,
        nheads: int = 8,
        dec_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.10,
        aux_pixel_weight: float = 0.40,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_pixel_weight = aux_pixel_weight

        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_dim, pixel_dim, 1, bias=False), nn.GroupNorm(8, pixel_dim), nn.GELU())
            for _ in range(4)
        ])

        self.mask_fuse = nn.Sequential(
            ConvGNAct(pixel_dim, pixel_dim, k=3, dropout=dropout),
            ConvGNAct(pixel_dim, pixel_dim, k=3, dropout=dropout),
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=1),
        )
        self.memory_fuse = nn.Sequential(
            ConvGNAct(pixel_dim * 3, pixel_dim, k=3, dropout=dropout),
            ConvGNAct(pixel_dim, pixel_dim, k=3, dropout=dropout),
        )
        self.aux_pixel_head = nn.Sequential(
            ConvGNAct(pixel_dim, pixel_dim, k=3, dropout=dropout),
            nn.Conv2d(pixel_dim, num_classes, kernel_size=1),
        )

        self.query_embed = nn.Embedding(num_queries, pixel_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=pixel_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.position_embed = PositionEmbeddingSine2D(num_pos_feats=pixel_dim // 2)

        self.class_embed = nn.Linear(pixel_dim, num_classes + 1)
        self.mask_embed = MLP(pixel_dim, pixel_dim, pixel_dim, num_layers=3)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = [proj(f) for proj, f in zip(self.proj, feats)]

        mask_feat = self.mask_fuse(p2)

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        memory_map = self.memory_fuse(torch.cat([p3, p4_up, p5_up], dim=1))

        b, d, h, w = memory_map.shape
        pos = self.position_embed(memory_map)
        memory = (memory_map + pos).flatten(2).transpose(1, 2).contiguous()

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.transformer_decoder(tgt=queries, memory=memory)

        class_logits = self.class_embed(hs)
        mask_embed = self.mask_embed(hs)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_feat)

        class_prob = F.softmax(class_logits, dim=-1)[..., :self.num_classes]
        mask_prob = torch.sigmoid(mask_logits)
        sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
        sem_logits = torch.log(sem_prob.clamp(min=1e-6, max=1.0))

        aux_logits = self.aux_pixel_head(mask_feat)
        sem_logits = sem_logits + self.aux_pixel_weight * aux_logits
        return sem_logits


class DINOv3SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: RawDINOv3Small16,
        num_classes: int = 2,
        return_layers: Sequence[int] = (2, 5, 8, 11),
        num_queries: int = 100,
    ):
        super().__init__()
        self.backbone = backbone
        self.return_layers = tuple(return_layers)
        self.adapter = ViTAdapterLite(vit_dim=384, adapter_dim=128, dropout=0.05)
        self.decoder = MultiScaleMask2FormerStyleHead(
            in_dim=128,
            pixel_dim=128,
            num_classes=num_classes,
            num_queries=num_queries,
            nheads=8,
            dec_layers=6,
            dim_feedforward=512,
            dropout=0.10,
            aux_pixel_weight=0.40,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_feats = self.backbone.forward_tokens(x, return_layers=self.return_layers)
        pyramid_feats = self.adapter(x, vit_feats)
        return self.decoder(pyramid_feats)


# =============================================================================
# Loss and metrics (与 stage2 完全一致)
# =============================================================================

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        target_safe = torch.where(valid, target, torch.zeros_like(target))
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        denom = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


@torch.no_grad()
def update_confusion_matrix(conf: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> None:
    mask = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if not np.any(mask):
        return
    t = target[mask].astype(np.int64)
    p = pred[mask].astype(np.int64)
    p = np.clip(p, 0, num_classes - 1)
    inds = num_classes * t + p
    cm = np.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    conf += cm


@torch.no_grad()
def compute_metrics_from_conf(conf: np.ndarray) -> Dict[str, object]:
    tp = np.diag(conf).astype(np.float64)
    row = conf.sum(axis=1).astype(np.float64)
    col = conf.sum(axis=0).astype(np.float64)
    union = row + col - tp
    iou = tp / np.maximum(union, 1.0)
    dice = 2.0 * tp / np.maximum(row + col, 1.0)
    acc = tp.sum() / max(conf.sum(), 1.0)
    valid_cls = np.ones_like(row, dtype=bool)
    miou = float(iou[valid_cls].mean()) if valid_cls.any() else 0.0
    mdice = float(dice[valid_cls].mean()) if valid_cls.any() else 0.0
    return {
        "pixel_acc": float(acc),
        "miou": miou,
        "mdice": mdice,
        "class_iou": [float(x) for x in iou.tolist()],
        "class_dice": [float(x) for x in dice.tolist()],
        "foreground_iou": float(iou[1]) if len(iou) > 1 else 0.0,
        "foreground_dice": float(dice[1]) if len(dice) > 1 else 0.0,
        "panicle_iou": float(iou[1]) if len(iou) > 1 else 0.0,
        "panicle_dice": float(dice[1]) if len(dice) > 1 else 0.0,
    }


def format_metrics(metrics: Dict[str, object]) -> str:
    return (
        f"mIoU={metrics['miou']:.5f}, mDice={metrics['mdice']:.5f}, "
        f"PixAcc={metrics['pixel_acc']:.5f}, "
        f"FgIoU={metrics['panicle_iou']:.5f}, FgDice={metrics['panicle_dice']:.5f}"
    )


# =============================================================================
# Train / Eval (与 stage2 完全一致)
# =============================================================================

def build_optimizer(model: DINOv3SegmentationModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = [
        {"params": backbone_params, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_backbone: float, lr_head: float) -> None:
    optimizer.param_groups[0]["lr"] = lr_backbone
    optimizer.param_groups[1]["lr"] = lr_head


def train_one_epoch(
    model: DINOv3SegmentationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_count = 0
    use_amp = cfg.amp == "fp16" and device.type == "cuda"

    pbar = tqdm(loader, desc=f"Train {epoch}/{cfg.epochs}", leave=False)
    for img, mask, _ in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(img)
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss_ce = ce_loss_fn(logits, mask)
            loss_dice = dice_loss_fn(logits, mask)
            loss = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = img.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_ce += float(loss_ce.detach().cpu()) * bs
        total_dice += float(loss_dice.detach().cpu()) * bs
        total_count += bs
        pbar.set_postfix({
            "loss": f"{float(loss.detach().cpu()):.4f}",
            "ce": f"{float(loss_ce.detach().cpu()):.4f}",
            "dice": f"{float(loss_dice.detach().cpu()):.4f}",
        })

    return {
        "loss": total_loss / max(total_count, 1),
        "ce": total_ce / max(total_count, 1),
        "dice_loss": total_dice / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(
    model: DINOv3SegmentationModel,
    loader: DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    desc: str = "Test",
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_count = 0
    conf = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    use_amp = cfg.amp == "fp16" and device.type == "cuda"

    for img, mask, _ in tqdm(loader, desc=desc, leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(img)
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss_ce = ce_loss_fn(logits, mask)
            loss_dice = dice_loss_fn(logits, mask)
            loss = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice
        pred = torch.argmax(logits, dim=1)
        bs = img.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_ce += float(loss_ce.detach().cpu()) * bs
        total_dice += float(loss_dice.detach().cpu()) * bs
        total_count += bs
        pred_np = pred.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        for p, t in zip(pred_np, mask_np):
            update_confusion_matrix(conf, p, t, cfg.num_classes, cfg.ignore_index)

    metrics = compute_metrics_from_conf(conf)
    metrics.update({
        "loss": total_loss / max(total_count, 1),
        "ce": total_ce / max(total_count, 1),
        "dice_loss": total_dice / max(total_count, 1),
        "confusion_matrix": conf.tolist(),
    })
    return metrics


def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, best_miou: float, best_panicle_iou: float, cfg: TrainConfig, history: List[dict]) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_miou": best_miou,
        "best_panicle_iou": best_panicle_iou,
        "config": asdict(cfg),
        "history": history,
        "class_names": CLASS_NAMES,
    }, path)


def resolve_resume_path(cfg: TrainConfig, ckpt_dir: str) -> Optional[str]:
    resume = cfg.resume
    if resume is None or str(resume).strip() == "" or str(resume).strip().lower() == "auto":
        auto_path = os.path.join(ckpt_dir, "last.pth")
        if os.path.exists(auto_path):
            print(f"[Resume] 自动发现 last checkpoint: {auto_path}")
            return auto_path
        print(f"[Resume] 未发现 last checkpoint: {auto_path}，将从头训练。")
        return None

    resume_str = str(resume).strip()
    if resume_str.lower() in {"none", "no", "false", "0"}:
        print("[Resume] 已通过 --resume none 禁用断点续训，将从头训练。")
        return None

    if not os.path.exists(resume_str):
        raise FileNotFoundError(f"指定的 resume checkpoint 不存在: {resume_str}")
    print(f"[Resume] 使用用户指定 checkpoint: {resume_str}")
    return resume_str


def load_resume(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], scaler: Optional[GradScaler], device: torch.device) -> Tuple[int, float, float, List[dict]]:
    print(f"[Resume] 加载: {path}")
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
        full_ckpt = True
    elif isinstance(ckpt, dict):
        model_state = ckpt
        full_ckpt = False
    else:
        raise RuntimeError(f"无法解析 checkpoint: {path}")

    model.load_state_dict(model_state, strict=True)
    print("[Resume] model.load_state_dict strict=True 成功")

    if full_ckpt and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("[Resume] optimizer 状态已恢复")
    else:
        print("[Resume] 未恢复 optimizer 状态；将使用新建 optimizer")

    if full_ckpt and scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
        print("[Resume] GradScaler 状态已恢复")
    else:
        print("[Resume] 未恢复 GradScaler 状态")

    if full_ckpt:
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("best_miou", 0.0))
        best_panicle_iou = float(ckpt.get("best_panicle_iou", 0.0))
        history = ckpt.get("history", [])
    else:
        start_epoch = 1
        best_miou = 0.0
        best_panicle_iou = 0.0
        history = []

    print(f"[Resume] start_epoch={start_epoch}, best_miou={best_miou:.5f}, best_panicle_iou={best_panicle_iou:.5f}")
    return start_epoch, best_miou, best_panicle_iou, history


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.output_dir)
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    save_json(asdict(cfg), os.path.join(cfg.output_dir, "config.json"))

    print("\n" + "=" * 96)
    print("Binary rice panicle segmentation: OFFICIAL DINOv3-S/16 .pth -> Mask2Former-style [direct stage2]")
    print("=" * 96)
    print(f"device        : {device}")
    print(f"backbone_ckpt : {cfg.backbone_ckpt}")
    print(f"train_rgb     : {cfg.train_rgb_dir}")
    print(f"train_label   : {cfg.train_label_dir}")
    print(f"test_rgb      : {cfg.test_rgb_dir}")
    print(f"test_label    : {cfg.test_label_dir}")
    print(f"output_dir    : {cfg.output_dir}")
    print(f"image_size    : {cfg.image_size}")
    print(f"batch_size    : {cfg.batch_size}")
    print(f"epochs        : {cfg.epochs}")
    print(f"lr_backbone   : {cfg.lr_backbone}")
    print(f"lr_head       : {cfg.lr_head}")
    print(f"freeze_epochs : {cfg.freeze_backbone_epochs}")
    print(f"val_ratio     : {cfg.val_ratio}")
    print(f"augment       : Albumentations={cfg.use_albumentations}, hflip={cfg.hflip_prob}, vflip={cfg.vflip_prob}, rot90={cfg.rotate90_prob}, color={cfg.color_jitter_prob}, blur={cfg.blur_prob}")
    print("classes       : " + ", ".join([f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)]))
    print("=" * 96 + "\n")

    all_pairs = build_pairs(cfg.train_rgb_dir, cfg.train_label_dir)
    all_pairs = validate_and_filter_pairs(all_pairs, cfg, "all-pairs")
    train_pairs, val_pairs = split_pairs(all_pairs, cfg.val_ratio, cfg.seed)
    print(f"[Dataset] train pairs: {len(train_pairs):,}; pseudo-val pairs: {len(val_pairs):,}")

    train_set = RiceSEGDataset(cfg.train_rgb_dir, cfg.train_label_dir, cfg, train=True, pairs=train_pairs, already_validated=True)
    test_set = RiceSEGDataset(cfg.test_rgb_dir, cfg.test_label_dir, cfg, train=False, pairs=val_pairs, already_validated=True)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    backbone = load_dinov3s_backbone(cfg.backbone_ckpt, image_size=cfg.image_size, device=device)
    model = DINOv3SegmentationModel(backbone, num_classes=cfg.num_classes).to(device)
    print(f"[Model] total params: {count_params(model)/1e6:.2f}M")
    print(f"[Model] backbone params: {count_params(model.backbone)/1e6:.2f}M")
    print(f"[Model] adapter+decoder params: {(count_params(model.adapter)+count_params(model.decoder))/1e6:.2f}M")

    # class weights
    if cfg.use_class_weights:
        class_weights = compute_class_weights(cfg.train_label_dir, cfg.num_classes, cfg.ignore_index, cfg.class_weight_max, pairs=train_pairs).to(device)
    else:
        class_weights = None
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index)
    dice_loss_fn = MultiClassDiceLoss(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index).to(device)

    optimizer = build_optimizer(model, cfg)
    scaler = GradScaler(enabled=(cfg.amp == "fp16" and device.type == "cuda"))

    start_epoch = 1
    best_miou = 0.0
    best_panicle_iou = 0.0
    history: List[dict] = []

    resume_path = resolve_resume_path(cfg, ckpt_dir)
    if resume_path is not None:
        start_epoch, best_miou, best_panicle_iou, history = load_resume(resume_path, model, optimizer, scaler, device)

        # 关键：如果上次已经训练到 cfg.epochs，直接 resume 会没有任何 epoch 可跑。
        # 因此默认自动延长 continue_epochs 个 epoch。
        if start_epoch > cfg.epochs and getattr(cfg, "continue_epochs", 0) > 0:
            old_epochs = cfg.epochs
            cfg.epochs = start_epoch + int(cfg.continue_epochs) - 1
            print(
                f"[Resume] checkpoint 已到达/超过原 epochs={old_epochs}，"
                f"自动延长 {cfg.continue_epochs} 个 epoch：继续训练到 epochs={cfg.epochs}"
            )

    if start_epoch > cfg.epochs:
        print(
            f"[Resume] start_epoch={start_epoch} > epochs={cfg.epochs}，没有需要继续训练的 epoch。\n"
            f"         解决方法：增大 --epochs，或设置 --continue_epochs 40。"
        )
        return

    print(f"[Train] 实际训练区间: epoch {start_epoch} -> {cfg.epochs}")

    history_path = os.path.join(cfg.output_dir, "history.jsonl")
    print("\n[Train] 开始训练，每个 epoch 结束后评估 pseudo-val 集。")
    print("-" * 140)
    print(f"{'Epoch':>5} │ {'TrainLoss':>10} │ {'ValLoss':>9} │ {'mIoU':>8} │ {'mDice':>8} │ {'FgIoU':>8} │ {'FgDice':>8} │ {'LR_b':>9} │ {'LR_h':>9} │ {'Best':>6}")
    print("-" * 140)

    for epoch in range(start_epoch, cfg.epochs + 1):
        # freeze / unfreeze backbone
        if epoch <= cfg.freeze_backbone_epochs:
            set_requires_grad(model.backbone, False)
            model.backbone.eval()
        else:
            set_requires_grad(model.backbone, True)

        min_lr_b = cfg.lr_backbone * cfg.min_lr_ratio
        min_lr_h = cfg.lr_head * cfg.min_lr_ratio
        lr_b = cosine_lr(cfg.lr_backbone, min_lr_b, epoch, cfg.epochs, cfg.warmup_epochs)
        lr_h = cosine_lr(cfg.lr_head, min_lr_h, epoch, cfg.epochs, cfg.warmup_epochs)
        if epoch <= cfg.freeze_backbone_epochs:
            lr_b = 0.0
        set_optimizer_lr(optimizer, lr_b, lr_h)

        train_result = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, dice_loss_fn, scaler, device, cfg, epoch)
        test_result = evaluate(model, test_loader, ce_loss_fn, dice_loss_fn, device, cfg, desc=f"PseudoVal {epoch}")

        is_best = False
        if float(test_result["miou"]) > best_miou:
            best_miou = float(test_result["miou"])
            is_best = True
            save_checkpoint(os.path.join(ckpt_dir, "best_miou.pth"), epoch, model, optimizer, scaler, best_miou, best_panicle_iou, cfg, history)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_miou_model_only.pth"))

        if float(test_result["panicle_iou"]) > best_panicle_iou:
            best_panicle_iou = float(test_result["panicle_iou"])
            save_checkpoint(os.path.join(ckpt_dir, "best_panicle_iou.pth"), epoch, model, optimizer, scaler, best_miou, best_panicle_iou, cfg, history)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_panicle_iou_model_only.pth"))

        record = {
            "epoch": epoch,
            "train": train_result,
            "test": test_result,
            "lr_backbone": lr_b,
            "lr_head": lr_h,
            "best_miou": best_miou,
            "best_panicle_iou": best_panicle_iou,
            "is_best_miou": is_best,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        history.append(record)
        append_jsonl(record, history_path)
        save_json(history, os.path.join(cfg.output_dir, "history.json"))

        save_checkpoint(os.path.join(ckpt_dir, "last.pth"), epoch, model, optimizer, scaler, best_miou, best_panicle_iou, cfg, history)
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"), epoch, model, optimizer, scaler, best_miou, best_panicle_iou, cfg, history)

        print(
            f"{epoch:5d} │ {train_result['loss']:10.5f} │ {test_result['loss']:9.5f} │ "
            f"{test_result['miou']:8.5f} │ {test_result['mdice']:8.5f} │ "
            f"{test_result['panicle_iou']:8.5f} │ {test_result['panicle_dice']:8.5f} │ "
            f"{lr_b:9.2e} │ {lr_h:9.2e} │ {'*' if is_best else '' :>6}"
        )

        # 保存每类 IoU 文本，方便直接看 前景类。
        per_class_path = os.path.join(cfg.output_dir, "per_class_last.txt")
        with open(per_class_path, "w", encoding="utf-8") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(format_metrics(test_result) + "\n\n")
            for i, name in enumerate(CLASS_NAMES):
                f.write(f"{i} {name}: IoU={test_result['class_iou'][i]:.6f}, Dice={test_result['class_dice'][i]:.6f}\n")

    print("\n[Done] 训练完成")
    print(f"Best mIoU: {best_miou:.6f}")
    print(f"Best foreground IoU: {best_panicle_iou:.6f}")
    print(f"输出目录: {cfg.output_dir}")


# =============================================================================
# Mask debug helper
# =============================================================================

def check_masks(label_dir: str, max_show: int = 20) -> None:
    """快速检查 mask 是否被正确读成二分类。"""
    files = [p for p in Path(label_dir).rglob("*") if p.is_file() and p.suffix.lower() in LABEL_EXTS]
    if not files:
        raise RuntimeError(f"mask 目录下没有找到标签文件: {label_dir}")

    total_pixels = 0
    total_fg = 0
    print("\n[MaskCheck] 检查 mask 读取与二值化结果")
    print(f"mask_dir: {label_dir}")
    print(f"files   : {len(files)}")
    print("-" * 100)
    print(f"{'file':50s} │ {'raw_unique[:10]':24s} │ {'fg_ratio':>10s}")
    print("-" * 100)

    for i, p in enumerate(files):
        arr = read_image_pil_or_cv2(str(p), mode="label")
        raw_unique = np.unique(arr)
        lab = normalize_label_array(arr, 2, 255, str(p))
        fg = int((lab == 1).sum())
        pixels = int((lab != 255).sum())
        total_fg += fg
        total_pixels += pixels
        if i < max_show:
            u = raw_unique[:10].astype(np.int64).tolist()
            print(f"{p.name[:50]:50s} │ {str(u):24s} │ {fg / max(pixels, 1):10.6f}")

    print("-" * 100)
    print(f"total_fg_pixels: {total_fg}")
    print(f"total_pixels   : {total_pixels}")
    print(f"total_fg_ratio : {total_fg / max(total_pixels, 1):.8f}")
    if total_fg == 0:
        print("\n[MaskCheck][ERROR] 前景像素为 0：训练一定会退化为全背景。")
        print('请确认 mask 目录是否正确；v3 已经按"非 0 为前景"处理，若仍为 0，说明这些 mask 文件本身全为 0 或不是标签文件。')
    else:
        print("\n[MaskCheck] OK：检测到前景像素，可以开始训练。")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("Binary rice panicle segmentation with DINOv3-S/16 backbone (official .pth direct stage2)")

    # 简化参数：只有一组 images/masks 时，直接传这两个；脚本会以 mask 为准配对，并自动切 pseudo-val。
    parser.add_argument("--image_dir", type=str, default=None, help="二分类训练图像目录；不传则使用 --train_rgb_dir")
    parser.add_argument("--mask_dir", type=str, default=None, help="二分类 mask 目录；不传则使用 --train_label_dir")

    parser.add_argument("--train_rgb_dir", type=str, default=TrainConfig.train_rgb_dir)
    parser.add_argument("--train_label_dir", type=str, default=TrainConfig.train_label_dir)
    parser.add_argument("--test_rgb_dir", type=str, default=TrainConfig.test_rgb_dir)
    parser.add_argument("--test_label_dir", type=str, default=TrainConfig.test_label_dir)
    parser.add_argument("--backbone_ckpt", type=str, default=TrainConfig.backbone_ckpt)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)

    parser.add_argument("--image_size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--patch_size", type=int, default=TrainConfig.patch_size)
    parser.add_argument("--num_classes", type=int, default=TrainConfig.num_classes)
    parser.add_argument("--ignore_index", type=int, default=TrainConfig.ignore_index)

    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)

    parser.add_argument("--lr_backbone", type=float, default=TrainConfig.lr_backbone)
    parser.add_argument("--lr_head", type=float, default=TrainConfig.lr_head)
    parser.add_argument("--min_lr_ratio", type=float, default=TrainConfig.min_lr_ratio)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--warmup_epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=TrainConfig.freeze_backbone_epochs)

    parser.add_argument("--amp", type=str, default=TrainConfig.amp, choices=["fp16", "none"])
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--ce_weight", type=float, default=TrainConfig.ce_weight)
    parser.add_argument("--dice_weight", type=float, default=TrainConfig.dice_weight)
    parser.add_argument("--no_class_weights", action="store_true", help="关闭类别权重")
    parser.add_argument("--class_weight_max", type=float, default=TrainConfig.class_weight_max)

    parser.add_argument("--hflip_prob", type=float, default=TrainConfig.hflip_prob)
    parser.add_argument("--vflip_prob", type=float, default=TrainConfig.vflip_prob)
    parser.add_argument("--rotate90_prob", type=float, default=TrainConfig.rotate90_prob)
    parser.add_argument("--color_jitter_prob", type=float, default=TrainConfig.color_jitter_prob)
    parser.add_argument("--blur_prob", type=float, default=TrainConfig.blur_prob)
    parser.add_argument("--no_albumentations", action="store_true", help="关闭 Albumentations 训练增强，仅做 resize/normalize")

    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--val_ratio", type=float, default=TrainConfig.val_ratio, help="从已配对伪标签样本中切分多少比例做 pseudo-val；0 表示 train/val 使用同一批。")
    parser.add_argument(
        "--resume",
        type=str,
        default=TrainConfig.resume,
        help="断点续训路径。默认 auto：自动加载 <output_dir>/checkpoints/last.pth；none 表示从头训练；也可传入具体 .pth 路径。",
    )
    parser.add_argument(
        "--continue_epochs",
        type=int,
        default=TrainConfig.continue_epochs,
        help="当自动续训发现 last.pth 已达到原 epochs 时，自动额外训练多少个 epoch。默认 40。",
    )
    parser.add_argument("--no_skip_bad_files", action="store_true", help="不跳过坏图/坏标签；遇到坏文件直接报错")
    parser.add_argument("--bad_file_log", type=str, default=TrainConfig.bad_file_log)
    parser.add_argument("--check_masks_only", action="store_true", help="只检查 mask 二值化结果，不训练")

    args = parser.parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k not in ["no_class_weights", "no_skip_bad_files", "no_albumentations", "image_dir", "mask_dir", "check_masks_only"]})
    if args.image_dir is not None:
        cfg.train_rgb_dir = args.image_dir
        cfg.test_rgb_dir = args.image_dir
    if args.mask_dir is not None:
        cfg.train_label_dir = args.mask_dir
        cfg.test_label_dir = args.mask_dir
    cfg.use_class_weights = not args.no_class_weights
    cfg.skip_bad_files = not args.no_skip_bad_files
    cfg.use_albumentations = not args.no_albumentations
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    # argparse 本身没有挂到 cfg 里；这里重新解析一次只取 check 标记，保持改动最小。
    parser_check = "--check_masks_only" in sys.argv
    if parser_check:
        check_masks(cfg.train_label_dir)
    else:
        train(cfg)
