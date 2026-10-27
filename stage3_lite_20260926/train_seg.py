"""Train/fine-tune RiceLite segmentation using explicit existing split manifests.

No automatic split or test-set checkpoint selection. JSON manifests are lists of
{"image": "path", "mask": "path"}; relative paths resolve against the manifest.
Teacher interface matches the archived Stage-2 source, loaded strictly.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from rice_lite import RiceLiteSeg, check_budget
from losses import Stage3Loss


def load_python(path):
    name = "rice_teacher_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclass introspection.
    spec.loader.exec_module(module)
    return module


def build_teacher(source, checkpoint, device):
    module = load_python(source)
    backbone = module.RawDINOv3Small16(image_size=512, patch_size=16, num_register_tokens=4)
    model = module.DINOv3SegmentationModel(backbone=backbone, num_classes=2,
                                          return_layers=(2, 5, 8, 11), num_queries=100)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


def build_mmseg_teacher(config, checkpoint, device):
    sys.path.insert(0, "/root/autodl-tmp/work/baseline_src/mmsegmentation")
    from mmengine.config import Config
    from mmseg.apis import init_model
    from mmseg.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(config)
    prep = cfg.model.data_preprocessor
    assert prep.get("bgr_to_rgb", False)
    assert np.allclose(prep.mean, np.array([.485, .456, .406]) * 255, atol=1e-3)
    assert np.allclose(prep.std, np.array([.229, .224, .225]) * 255, atol=1e-3)
    model = init_model(cfg, checkpoint=None, device="cpu")
    # Trusted existing experiment checkpoint contains MMSeg metadata.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model.requires_grad_(False).eval().to(device)


@torch.no_grad()
def teacher_forward(model, images):
    if hasattr(model, "decode_head"):
        features = model.extract_feat(images)
        return {"features": features, "logits": model.decode_head(features)}
    tokens = model.backbone.forward_tokens(images, return_layers=model.return_layers)
    features = model.adapter(images, tokens)
    return {"features": features, "logits": model.decoder(features)}


class SegDataset(Dataset):
    def __init__(self, manifest, size=512, training=False, encoding="indices"):
        manifest = Path(manifest).resolve()
        rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest must contain a nonempty list")
        self.rows = []
        for row in rows:
            paths = tuple((manifest.parent / row[key]).resolve() for key in ("image", "mask"))
            if not all(p.is_file() for p in paths):
                raise FileNotFoundError(paths)
            self.rows.append(paths)
        if len(set(p[0] for p in self.rows)) != len(self.rows):
            raise ValueError("Duplicate images in manifest")
        self.size, self.training, self.encoding = size, training, encoding

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, mask_path = self.rows[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            original_size = image.size
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(mask_path) as source:
            if source.size != original_size:
                raise ValueError(f"Image/mask dimensions differ: {image_path}, {mask_path}")
            if self.encoding == "rgb-nonzero":
                source = source.convert("RGB")
            mask = np.array(source.resize((self.size, self.size), Image.Resampling.NEAREST))
        if self.encoding == "rgb-nonzero":
            mask = np.any(mask != 0, axis=2).astype(np.int64)
        elif mask.ndim != 2:
            raise ValueError(f"Use an explicit RGB mask encoding for {mask_path}")
        elif self.encoding == "binary255":
            if not np.isin(mask, [0, 255]).all():
                raise ValueError(f"Expected only 0/255: {mask_path}")
            mask = (mask == 255).astype(np.int64)
        elif not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"Expected class indices 0/1 and ignore=255: {mask_path}")
        rgb = np.asarray(image, dtype=np.float32) / 255
        if self.training and random.random() < 0.5:
            rgb, mask = rgb[:, ::-1], mask[:, ::-1]
        if self.training and random.random() < 0.25:
            rgb, mask = rgb[::-1], mask[::-1]
        # Teacher and student receive exactly the same augmented RGB tensor.
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1), torch.from_numpy(mask.astype(np.int64).copy())


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    matrix = torch.zeros(2, 2, dtype=torch.int64, device=device)
    for images, target in loader:
        target = target.to(device)
        pred = model(images.to(device))["logits"].argmax(1)
        valid = target != 255
        matrix += torch.bincount(2 * target[valid] + pred[valid], minlength=4).reshape(2, 2)
    matrix = matrix.cpu().double()
    union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
    present = union > 0
    if not present.any():
        raise ValueError("Validation set has no valid pixels")
    iou = matrix.diag() / union.clamp_min(1)
    return {"miou": iou[present].mean().item(), "panicle_iou": iou[1].item(),
            "confusion_matrix": matrix.long().tolist()}


def train(args):
    if args.epochs < 1 or args.size < 32 or args.batch_size < 1:
        raise ValueError("Invalid epochs/image size/batch size")
    teacher_config = getattr(args, "teacher_config", None)
    resume = getattr(args, "resume", None)
    if args.teacher_source and teacher_config:
        raise ValueError("Choose only one teacher source/config")
    if bool(args.teacher_source or teacher_config) != bool(args.teacher_weights):
        raise ValueError("Provide teacher source/config AND weights")
    if resume and args.init:
        raise ValueError("Choose --resume or --init")
    if args.teacher_weights and args.size != 512:
        raise ValueError("Archived Stage-2 teacher integration uses its original 512 input")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_data = SegDataset(args.train_manifest, args.size, True, args.mask_encoding)
    val_data = SegDataset(args.val_manifest, args.size, False, args.mask_encoding)
    if set(p[0] for p in train_data.rows) & set(p[0] for p in val_data.rows):
        raise ValueError("Train/validation image paths overlap")
    if set(p[1] for p in train_data.rows) & set(p[1] for p in val_data.rows):
        raise ValueError("Train/validation mask paths overlap")
    # Overlapping crops from one original image must still be grouped upstream.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers,
                              worker_init_fn=seed_worker, generator=generator,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, args.batch_size, num_workers=args.workers)
    model = RiceLiteSeg(args.variant, context=not args.no_context).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
    teacher = None
    if args.teacher_weights:
        builder = build_mmseg_teacher if teacher_config else build_teacher
        teacher = builder(teacher_config or args.teacher_source, args.teacher_weights, device)
    criterion = Stage3Loss(kd_weight=args.kd_weight, feature_weight=args.feature_weight,
                           boundary_weight=args.boundary_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=bool(resume))
    settings = vars(args).copy()
    settings.update(parameters=check_budget(model), teacher_loaded_strictly=teacher is not None,
                    counts={"train": len(train_data), "val": len(val_data)},
                    manifest_sha256={key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
                                     for key in ("train_manifest", "val_manifest")})
    history, best, first_epoch = [], -1.0, 0
    if resume:
        previous = json.loads((output / "config.json").read_text())
        for key in ("variant", "no_context", "epochs", "seed", "manifest_sha256", "lr",
                    "batch_size", "kd_weight", "feature_weight", "boundary_weight",
                    "teacher_source", "teacher_config", "teacher_weights", "mask_encoding", "select_by"):
            if previous.get(key) != settings.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, best, history = state["epoch"], state["best"], state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps({"event": "ready", "parameters": settings["parameters"],
                      "teacher": teacher is not None, "counts": settings["counts"],
                      "first_epoch": first_epoch + 1}), flush=True)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(5 * len(train_loader), max(1, total_steps // 10))
    started = time.time()
    for epoch in range(first_epoch, args.epochs):
        epoch_started = time.time()
        model.train()
        sums = {}
        seen = 0
        for iteration, (images, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + iteration
            if step < warmup_steps:
                multiplier = (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                multiplier = .01 + .99 * .5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Teacher runs FP32, as in the archived Stage-3 training recipe.
            teacher_output = teacher_forward(teacher, images) if teacher is not None else None
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = criterion(model(images), target, teacher_output,
                                   progress=(step + 1) / total_steps)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + value.detach().item() * images.shape[0]
            seen += images.shape[0]
            if iteration == 0 or (iteration + 1) % 25 == 0:
                print(json.dumps({"event": "step", "epoch": epoch + 1,
                                  "step": iteration + 1, "steps": len(train_loader),
                                  "loss": losses["total"].detach().item(),
                                  "lr": optimizer.param_groups[0]["lr"],
                                  "seconds": time.time() - epoch_started}), flush=True)
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "loss": {k: v / seen for k, v in sums.items()},
                  "epoch_seconds": time.time() - epoch_started,
                  "elapsed_seconds": time.time() - started, **metrics}
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = {"model": model.state_dict(), "variant": args.variant,
                      "context": not args.no_context, "epoch": epoch + 1, "validation": metrics}
        if metrics[args.select_by] > best:
            best = metrics[args.select_by]
            torch.save(checkpoint, output / "best.pt")
        torch.save(checkpoint, output / "last.pt")
        resumable = {**checkpoint, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "best": best, "history": history, "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "loader_rng": generator.get_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        torch.save(resumable, output / "resume.tmp")
        (output / "resume.tmp").replace(output / "resume.pt")
        print(json.dumps(record), flush=True)
    return history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--variant", choices=("base", "micro"), default="base")
    p.add_argument("--teacher-source", help="Archived Stage-2 .py (original model, not MMSeg wrapper)")
    p.add_argument("--teacher-config", help="Existing MMSeg teacher config, alternative to source")
    p.add_argument("--teacher-weights", help="Matching original Stage-2 full model checkpoint")
    p.add_argument("--init", help="RiceLite unfused checkpoint for a new fine-tuning run")
    p.add_argument("--resume", help="Resume an interrupted run from its resume.pt")
    p.add_argument("--mask-encoding", choices=("indices", "binary255", "rgb-nonzero"), default="indices")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=0.5)
    p.add_argument("--boundary-weight", type=float, default=0.2)
    p.add_argument("--select-by", choices=("miou", "panicle_iou"), default="miou")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
