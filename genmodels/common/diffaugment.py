"""DiffAugment (Zhao et al., 2020): differentiable augmentation for data-efficient GANs.

With a few hundred images a GAN discriminator memorises the training set within
minutes and stops giving the generator useful gradients. DiffAugment applies the
*same kind* of random augmentation to both real and fake images every time they
are shown to D. Because the augmentations are differentiable, G still gets
gradients through them, and because they are applied to both sides, G is not
pushed to generate augmented-looking images.

Policies (comma-separated): ``color``, ``translation``, ``cutout``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _color(x):
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)  # brightness
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=x.device) * 2) + mean  # saturation
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(b, 1, 1, 1, device=x.device) + 0.5) + mean  # contrast


def _translation(x, ratio=0.125):
    b, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gx = (gx + tx + 1).clamp(0, h + 1)
    gy = (gy + ty + 1).clamp(0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    return padded.permute(0, 2, 3, 1)[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio=0.5):
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = (gx + ox - ch // 2).clamp(0, h - 1)
    gy = (gy + oy - cw // 2).clamp(0, w - 1)
    mask = torch.ones(b, h, w, device=x.device, dtype=x.dtype)
    mask[gb, gx, gy] = 0
    return x * mask[:, None]


_OPS = {"color": _color, "translation": _translation, "cutout": _cutout}


def diff_augment(x: torch.Tensor, policy: str | None) -> torch.Tensor:
    if not policy:
        return x
    for name in policy.split(","):
        x = _OPS[name.strip()](x)
    return x
