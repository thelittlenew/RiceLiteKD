"""Resolution-normalized, ignore-safe binary segmentation distillation."""
import torch
from torch import nn
from torch.nn import functional as F


class Stage3Loss(nn.Module):
    def __init__(self, temperature=2.0, confidence=0.7,
                 kd_weight=1.0, feature_weight=0.5, boundary_weight=0.2):
        super().__init__()
        if temperature <= 0 or not 0 <= confidence <= 1:
            raise ValueError("Invalid temperature/confidence")
        self.temperature, self.confidence = temperature, confidence
        self.kd_weight, self.feature_weight = kd_weight, feature_weight
        self.boundary_weight = boundary_weight

    @staticmethod
    def balanced_mean(values, labels, valid):
        # Equal foreground/background contribution among classes present.
        terms = []
        for class_id in (0, 1):
            mask = valid & (labels == class_id)
            if mask.any():
                terms.append(values[mask].mean())
        return torch.stack(terms).mean() if terms else values.sum() * 0

    def forward(self, output, target, teacher=None, progress=1.0):
        logits = output["logits"].float()
        if logits.shape[1] != 2:
            raise ValueError("Stage3Loss expects two classes")
        valid = target != 255
        safe = target.masked_fill(~valid, 0)
        ce = self.balanced_mean(F.cross_entropy(logits, safe, reduction="none"),
                                safe, valid)
        prob = logits.softmax(1)[:, 1]
        truth = (safe == 1).float()
        intersection = (prob * truth * valid).sum((1, 2))
        denominator = ((prob + truth) * valid).sum((1, 2))
        dice_per_image = 1 - (2 * intersection + 1) / (denominator + 1)
        present = valid.flatten(1).any(1)
        dice = dice_per_image[present].mean() if present.any() else logits.sum() * 0
        # Neighbour differences supervise edges without an inference edge branch.
        boundary_terms = []
        for dim in (1, 2):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[dim], b[dim] = slice(1, None), slice(None, -1)
            a, b = tuple(a), tuple(b)
            edge_valid = valid[a] & valid[b]
            pred_edge = (prob[a] - prob[b]).abs()
            true_edge = (truth[a] - truth[b]).abs()
            boundary_terms.append(self.balanced_mean((pred_edge - true_edge).abs(),
                                                      true_edge.long(), edge_valid))
        boundary = torch.stack(boundary_terms).mean()
        kd, feature = logits.sum() * 0, logits.sum() * 0
        if teacher is not None:
            tlogits = F.interpolate(teacher["logits"].detach().float(),
                                   size=logits.shape[-2:], mode="bilinear",
                                   align_corners=False)
            confidence, pseudo = tlogits.softmax(1).max(1)
            trusted = valid & (confidence >= self.confidence)
            temperature = self.temperature
            per_pixel = F.kl_div(F.log_softmax(logits / temperature, 1),
                                 F.softmax(tlogits / temperature, 1),
                                 reduction="none").sum(1) * temperature ** 2
            kd = self.balanced_mean(per_pixel, pseudo, trusted)
            if len(output["features"]) != 4 or len(teacher["features"]) != 4:
                raise ValueError("Expected P2/P3/P4/P5 features from both models")
            # Channel-independent spatial attention transfer on P3/P4/P5.
            # Student need not carry teacher-width projections in deployment.
            terms = []
            for sf, tf in zip(output["features"][1:], teacher["features"][1:]):
                sa = sf.float().square().mean(1, keepdim=True)
                ta = tf.detach().float().square().mean(1, keepdim=True)
                ta = F.interpolate(ta, size=sa.shape[-2:], mode="bilinear", align_corners=False)
                mask = F.interpolate(valid[:, None].float(), size=sa.shape[-2:], mode="nearest")
                sa = F.normalize((sa * mask).flatten(1), dim=1, eps=1e-6)
                ta = F.normalize((ta * mask).flatten(1), dim=1, eps=1e-6)
                per_image = (sa - ta).square().sum(1)
                keep = mask.flatten(1).any(1)
                terms.append(per_image[keep].mean() if keep.any() else sa.sum() * 0)
            feature = torch.stack(terms).mean()
        ramp = min(max(float(progress), 0.0) / 0.1, 1.0)
        total = ce + dice + self.boundary_weight * boundary + ramp * (
            self.kd_weight * kd + self.feature_weight * feature)
        return {"total": total, "ce": ce, "dice": dice, "boundary": boundary,
                "kd": kd, "feature": feature}
