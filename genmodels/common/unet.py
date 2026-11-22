"""The U-Net used by both diffusion and flow matching (after Ho et al., 2020).

It maps (noisy image x_t, time t, optional class y) -> a tensor shaped like x_t:
the predicted noise for diffusion, the velocity for flow matching. ``t`` is a
float per sample; diffusion passes integer timesteps 0..T-1, flow matching
passes t in [0, 1] scaled by 1000, so both see the same embedding range.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of a (B,) float tensor, like transformer positional encodings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def _norm(c):
    return nn.GroupNorm(min(32, c // 4) if c >= 4 else 1, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1, self.conv1 = _norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout * 2)  # scale and shift from the time/class embedding
        self.norm2, self.conv2 = _norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = _norm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, in_channels=3, image_size=32, base_channels=64, channel_mults=(1, 2, 2),
                 num_res_blocks=2, attn_resolutions=(16,), dropout=0.1, num_classes=0):
        super().__init__()
        emb_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # Class embedding with one extra "null" class (index num_classes) for
        # classifier-free guidance: training sometimes drops the label to it.
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim) if num_classes else None

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        skips, ch, res = [base_channels], base_channels, image_size
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                out = base_channels * mult
                self.down.append(nn.ModuleList([ResBlock(ch, out, emb_dim, dropout),
                                                Attention(out) if res in attn_resolutions else nn.Identity()]))
                ch = out
                skips.append(ch)
            if i != len(channel_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skips.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(num_res_blocks + 1):
                out = base_channels * mult
                self.up.append(nn.ModuleList([ResBlock(ch + skips.pop(), out, emb_dim, dropout),
                                              Attention(out) if res in attn_resolutions else nn.Identity()]))
                ch = out
            if i != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(ch, ch, 3, padding=1)]))
                res *= 2

        self.out = nn.Sequential(_norm(ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y=None):
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        if self.class_emb is not None:
            if y is None:
                y = torch.full((len(x),), self.num_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)

        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            if len(layer) == 2:
                h = layer[1](layer[0](h, emb))
            else:
                h = layer[0](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer[0], ResBlock):
                h = layer[1](layer[0](torch.cat([h, hs.pop()], dim=1), emb))
            else:
                h = layer[1](layer[0](h))
        return self.out(h)
