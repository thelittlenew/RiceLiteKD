"""RiceLite Stage 3 prototypes. Pure PyTorch; no pretrained accuracy claimed.

Design sources and departures from them are documented in README.md.
Parameter cap applies to the COMPLETE unfused model, including its task head.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


CONFIGS = {
    "micro": ((16, 32, 64, 112), (1, 2, 3, 3), 24),
    "base": ((24, 48, 96, 160), (2, 3, 4, 4), 32),
}


def parameter_count(model):
    # Includes frozen parameters; requires_grad is NOT a size criterion.
    return sum(p.numel() for p in model.parameters())


def check_budget(model, limit=1_000_000):
    count = parameter_count(model)
    if count > limit:
        raise ValueError(f"Complete model has {count:,} parameters > {limit:,}")
    return count


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False) if act else nn.Identity(),
        )

    def fuse(self):
        return nn.Sequential(torch.nn.utils.fusion.fuse_conv_bn_eval(self[0], self[1]),
                             deepcopy(self[2]))


class RepDepthwise(nn.Module):
    """Linear kxk DW + 1x1 DW + identity; merged into ONE DW convolution."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.large = ConvBN(channels, channels, kernel, groups=channels, act=False)
        self.small = ConvBN(channels, channels, 1, groups=channels, act=False)

    def forward(self, x):
        return self.large(x) + self.small(x) + x

    @torch.no_grad()
    def fuse(self):
        large, small = self.large.fuse()[0], self.small.fuse()[0]
        pad = large.kernel_size[0] // 2
        large.weight.add_(F.pad(small.weight, (pad, pad, pad, pad)))
        large.weight[:, 0, pad, pad].add_(1)
        large.bias.add_(small.bias)
        return large


class LiteBlock(nn.Module):
    """Spatial mixing BEFORE 2x pointwise expansion to limit activation cost."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.spatial = RepDepthwise(channels, kernel)
        self.channel = nn.Sequential(ConvBN(channels, 2 * channels),
                                     ConvBN(2 * channels, channels, act=False))

    def forward(self, x):
        y = self.spatial(x)
        return y + self.channel(y)


class Downsample(nn.Sequential):
    """Reduce spatial size before expanding channels (UIB-inspired choice)."""
    def __init__(self, cin, cout):
        super().__init__(ConvBN(cin, cin, 3, stride=2, groups=cin, act=False),
                         ConvBN(cin, 2 * cin),
                         ConvBN(2 * cin, cout, act=False))


class BottleneckContext(nn.Module):
    """One custom single-head attention: Q at /32, K/V capped at 8x8.

    Inspired by low-resolution context modeling; NOT a TinyNeXt reproduction.
    This has O(HW * 64) attention storage, instead of full-resolution N squared.
    """
    def __init__(self, channels, key_dim=16, value_dim=32):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.q = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.k = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.v = nn.Conv2d(channels, value_dim, 1, bias=False)
        self.proj = ConvBN(value_dim, channels, act=False)
        self.scale = key_dim ** -0.5

    def forward(self, x):
        z = self.norm(x)
        h, w = z.shape[-2:]
        pooled = F.adaptive_avg_pool2d(z, (min(h, 8), min(w, 8)))
        q = self.q(z).flatten(2).transpose(1, 2)
        k = self.k(pooled).flatten(2)
        v = self.v(pooled).flatten(2).transpose(1, 2)
        # FP32 softmax avoids overflow under AMP; cast only the probabilities.
        attention = (q.float() @ k.float() * self.scale).softmax(-1).to(v.dtype)
        y = (attention @ v).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return x + self.proj(y)


class RiceLiteBackbone(nn.Module):
    def __init__(self, variant="base", context=True):
        super().__init__()
        widths, depths, _ = CONFIGS[variant]
        self.out_channels = widths
        self.stem = ConvBN(3, 16, 3, stride=2)
        self.stages = nn.ModuleList()
        cin = 16
        for index, (width, depth) in enumerate(zip(widths, depths)):
            self.stages.append(nn.Sequential(
                Downsample(cin, width),
                *[LiteBlock(width, 3 if index < 2 else 5) for _ in range(depth)],
            ))
            cin = width
        self.context = BottleneckContext(widths[-1]) if context else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        features[-1] = self.context(features[-1])
        return features


class Separable(nn.Sequential):
    def __init__(self, width):
        super().__init__(ConvBN(width, width, 3, groups=width), ConvBN(width, width))


class AdditivePyramid(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.lateral = nn.ModuleList([ConvBN(c, width) for c in in_channels])
        self.refine = nn.ModuleList([Separable(width) for _ in in_channels])

    def forward(self, features):
        out = [projection(x) for projection, x in zip(self.lateral, features)]
        out[-1] = self.refine[-1](out[-1])
        for i in range(len(out) - 2, -1, -1):
            y = F.interpolate(out[i + 1], size=out[i].shape[-2:], mode="nearest")
            out[i] = self.refine[i](out[i] + y)
        return out


class RiceLiteSeg(nn.Module):
    def __init__(self, variant="base", num_classes=2, context=True):
        super().__init__()
        self.backbone = RiceLiteBackbone(variant, context)
        width = CONFIGS[variant][2]
        self.decoder = AdditivePyramid(self.backbone.out_channels, width)
        self.classifier = nn.Conv2d(width, num_classes, 1)
        check_budget(self)

    def forward(self, images):
        features = self.backbone(images)
        decoded = self.decoder(features)
        low_logits = self.classifier(decoded[0])
        # Upsample 2 logits, never a 128-channel full-resolution activation.
        logits = F.interpolate(low_logits, size=images.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"logits": logits, "features": features}


def to_deploy(model):
    """Non-mutating eval-only fusion. Save original unfused state for training."""
    if model.training:
        raise ValueError("Call eval() before deployment fusion")
    fused = deepcopy(model)
    def convert(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (RepDepthwise, ConvBN)):
                setattr(module, name, child.fuse())
            else:
                convert(child)
    convert(fused)
    check_budget(fused)
    return fused
