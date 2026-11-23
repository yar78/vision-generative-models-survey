"""A compact StyleGAN2 (Karras et al., 2019/2020), sized for 32-64 px experiments.

What makes it different from a DCGAN generator:

* **Mapping network** f: z -> w. An 8-layer MLP turns Gaussian noise into an
  intermediate latent w that does not have to follow a fixed distribution, which
  makes w far more disentangled than z.
* **Constant input.** Synthesis starts from a learned 4x4 tensor, not from z.
* **Modulated convolutions.** Each layer's conv weights are scaled per input
  channel by a "style" s = A(w), then *demodulated* (renormalised) so activations
  keep unit variance. This replaces StyleGAN1's AdaIN.
* **Noise inputs** add per-pixel stochastic detail (hair, texture) at each layer.
* **Skip-connection toRGB outputs** summed across resolutions.
* **Style mixing**: during training, some layers get w from a second z, so
  layers learn to control separable aspects (coarse layers = pose/shape, fine
  layers = colour/texture).
* **Truncation trick** at sampling: w' = w_avg + psi * (w - w_avg) trades
  diversity for quality.

Training uses the non-saturating logistic loss with lazy R1 regularisation on
real images, and optionally DiffAugment for small datasets. Omitted for size: path-length regularisation, adaptive
augmentation (ADA), and the FIR-filtered up/downsampling of the paper (bilinear
resampling is used instead).
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .common.diffaugment import diff_augment
from .common.trainer import BaseTrainer

SQRT2 = math.sqrt(2)


class EqualLinear(nn.Module):
    """Linear layer with the *equalised learning rate* trick: weights are stored as
    N(0,1) and scaled at runtime, so every layer learns at the same speed."""

    def __init__(self, cin, cout, bias_init=0.0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin) / lr_mul)
        self.bias = nn.Parameter(torch.full((cout,), float(bias_init)))
        self.scale = lr_mul / math.sqrt(cin)
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class EqualConv2d(nn.Module):
    """Conv2d with the equalised learning rate: N(0,1) weights scaled by 1/sqrt(fan_in) at
    runtime. StyleGAN's learning rate (~2e-3) assumes this in *both* networks; with plain
    PyTorch layers the discriminator takes steps 10-100x too large and its logits explode."""

    def __init__(self, cin, cout, kernel, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None
        self.scale = 1 / math.sqrt(cin * kernel ** 2)
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, n_layers=8, lr_mul=0.01, w_avg_beta=0.995):
        super().__init__()
        layers = [EqualLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=lr_mul) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.w_avg_beta = w_avg_beta
        self.register_buffer("w_avg", torch.zeros(w_dim))  # for the truncation trick

    def forward(self, z):
        x = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # pixel norm
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2) * SQRT2
        if self.training:
            self.w_avg.copy_(x.detach().mean(0).lerp(self.w_avg, self.w_avg_beta))
        return x


class ModulatedConv2d(nn.Module):
    """Conv whose weights are modulated per sample by a style vector, then demodulated."""

    def __init__(self, cin, cout, kernel, w_dim, demodulate=True, upsample=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, cout, cin, kernel, kernel))
        self.scale = 1 / math.sqrt(cin * kernel ** 2)
        self.affine = EqualLinear(w_dim, cin, bias_init=1.0)  # A: w -> style s
        self.demodulate, self.upsample, self.padding = demodulate, upsample, kernel // 2
        self.cout = cout

    def forward(self, x, w):
        b, cin, h, wd = x.shape
        s = self.affine(w)                                            # (B, cin)
        weight = self.scale * self.weight * s[:, None, :, None, None]  # modulate
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)   # (B, cout)
            weight = weight * d[:, :, None, None, None]
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h, wd = h * 2, wd * 2
        # One grouped conv applies a different kernel to every sample in the batch.
        x = x.reshape(1, b * cin, h, wd)
        weight = weight.reshape(b * self.cout, cin, *weight.shape[3:])
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.reshape(b, self.cout, h, wd)


class StyleLayer(nn.Module):
    def __init__(self, cin, cout, w_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(cin, cout, 3, w_dim, upsample=upsample)
        self.noise_strength = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, w, noise=None):
        x = self.conv(x, w)
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_strength * noise
        return F.leaky_relu(x + self.bias[None, :, None, None], 0.2) * SQRT2


class ToRGB(nn.Module):
    def __init__(self, cin, out_channels, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(cin, out_channels, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w):
        return self.conv(x, w) + self.bias[None, :, None, None]


class StyleGANGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, out_channels: int = 3, image_size: int = 32, z_dim: int = 256, w_dim: int = 256,
                 mapping_layers: int = 8, channels: int = 256, max_channels: int = 256):
        super().__init__()
        self.z_dim, self.w_dim = z_dim, w_dim
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        n_up = int(math.log2(image_size)) - 2
        chans = [min(max_channels, channels // 2 ** max(0, i - 1)) for i in range(n_up + 1)]  # per resolution

        self.const = nn.Parameter(torch.randn(1, chans[0], 4, 4))
        self.layers = nn.ModuleList([StyleLayer(chans[0], chans[0], w_dim)])
        self.to_rgbs = nn.ModuleList([ToRGB(chans[0], out_channels, w_dim)])
        for i in range(1, n_up + 1):
            self.layers.append(StyleLayer(chans[i - 1], chans[i], w_dim, upsample=True))
            self.layers.append(StyleLayer(chans[i], chans[i], w_dim))
            self.to_rgbs.append(ToRGB(chans[i], out_channels, w_dim))
        # One w per conv layer plus one per toRGB; style mixing swaps a suffix of these.
        self.num_ws = len(self.layers) + len(self.to_rgbs)

    def map(self, z, truncation: float = 1.0):
        w = self.mapping(z)
        if truncation != 1.0:
            w = self.mapping.w_avg.lerp(w, truncation)
        return w

    def synthesis(self, ws):
        """ws: (B, num_ws, w_dim). Layer order: conv0, rgb0, [up-conv, conv, rgb] per resolution."""
        x = self.const.expand(len(ws), -1, -1, -1)
        x = self.layers[0](x, ws[:, 0])
        rgb = self.to_rgbs[0](x, ws[:, 1])
        k = 2
        for i in range(1, len(self.to_rgbs)):
            x = self.layers[2 * i - 1](x, ws[:, k])
            x = self.layers[2 * i](x, ws[:, k + 1])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False) + self.to_rgbs[i](x, ws[:, k + 2])
            k += 3
        return rgb

    def forward(self, z, truncation: float = 1.0, mix_z=None, mix_cutoff: int | None = None):
        w = self.map(z, truncation)
        ws = w[:, None].repeat(1, self.num_ws, 1)
        if mix_z is not None:
            cutoff = mix_cutoff if mix_cutoff is not None else random.randint(1, self.num_ws - 1)
            ws[:, cutoff:] = self.map(mix_z, truncation)[:, None]
        return self.synthesis(ws)

    @torch.no_grad()
    def generate(self, n: int, truncation: float = 1.0, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        # No tanh at the output (as in StyleGAN2): the raw image is clamped for display.
        return self(torch.randn(n, self.z_dim, device=device), truncation=truncation).clamp(-1, 1)


class DiscBlock(nn.Module):
    """Residual downsampling block."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = EqualConv2d(cin, cin, 3, padding=1)
        self.conv2 = EqualConv2d(cin, cout, 3, padding=1)
        self.skip = EqualConv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.avg_pool2d(F.leaky_relu(self.conv2(h), 0.2), 2)
        return (h + F.avg_pool2d(self.skip(x), 2)) / SQRT2


class StyleGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, image_size=32, channels=256, max_channels=256, mbstd_group=4):
        super().__init__()
        n_down = int(math.log2(image_size)) - 2
        chans = [min(max_channels, channels // 2 ** max(0, i - 1)) for i in range(n_down + 1)][::-1]
        self.from_rgb = EqualConv2d(in_channels, chans[0], 1)
        self.blocks = nn.Sequential(*[DiscBlock(chans[i], chans[i + 1]) for i in range(n_down)])
        self.mbstd_group = mbstd_group
        c = chans[-1]
        self.conv = EqualConv2d(c + 1, c, 3, padding=1)
        self.fc = EqualLinear(c * 16, c)
        self.out = EqualLinear(c, 1)

    def minibatch_std(self, x):
        """Append the across-batch std as a feature map, so D can spot low diversity."""
        b, c, h, w = x.shape
        g = min(self.mbstd_group, b)
        while b % g:
            g -= 1
        y = x.view(g, -1, c, h, w)
        y = (y.var(dim=0, unbiased=False) + 1e-8).sqrt().mean(dim=(1, 2, 3))  # (b/g,)
        y = y.view(-1, 1, 1, 1).repeat(g, 1, h, w)
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        x = F.leaky_relu(self.from_rgb(x), 0.2)
        x = self.blocks(x)
        x = F.leaky_relu(self.conv(self.minibatch_std(x)), 0.2)
        x = F.leaky_relu(self.fc(x.flatten(1)), 0.2)
        return self.out(x).view(-1)


def r1_penalty(D, real):
    """R1: ||grad_x D(x)||^2 on real images -- penalises D for being steep around the data."""
    real = real.detach().requires_grad_(True)
    grad, = torch.autograd.grad(D(real).sum(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


class Trainer(BaseTrainer):
    family = "stylegan"

    def build(self):
        m = dict(self.model_cfg)
        d_channels = m.pop("d_channels", m.get("channels", 256))
        self.G = StyleGANGenerator(self.channels, self.image_size, **m).to(self.device)
        self.D = StyleGANDiscriminator(self.channels, self.image_size, d_channels,
                                       max_channels=m.get("max_channels", 256)).to(self.device)
        self.nets = {"G": self.G, "D": self.D}
        lr = float(self.train_cfg.get("lr", 2.5e-3))
        self.r1_gamma = float(self.method_cfg.get("r1_gamma", 0.05))
        self.r1_every = int(self.method_cfg.get("r1_every", 16))
        # Lazy regularisation (StyleGAN2 appendix B): R1 runs every r1_every steps, so D's
        # optimizer is adjusted as if the penalty were spread over every step.
        c = self.r1_every / (self.r1_every + 1)
        self.opts = {"G": torch.optim.Adam(self.G.parameters(), lr, betas=(0.0, 0.99)),
                     "D": torch.optim.Adam(self.D.parameters(), lr * c, betas=(0.0, 0.99 ** c))}
        self.add_ema("G")
        self.mixing_prob = float(self.method_cfg.get("style_mixing_prob", 0.9))
        self.augment = self.method_cfg.get("diffaugment")  # e.g. "color,translation,cutout" for small datasets

    def D_aug(self, x):
        """D sees real and fake images through the same random differentiable augmentation."""
        return self.D(diff_augment(x, self.augment))

    def _fake(self, n):
        z = torch.randn(n, self.G.z_dim, device=self.device)
        mix = torch.randn_like(z) if random.random() < self.mixing_prob else None
        return self.G(z, mix_z=mix)

    def training_step(self, x, y):
        # --- D: non-saturating logistic loss, plus lazy R1 every r1_every steps -----
        with torch.no_grad():
            fake = self._fake(len(x))
        d_real, d_fake = self.D_aug(x), self.D_aug(fake)
        d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
        logs = {}
        if self.step % self.r1_every == 0:
            r1 = r1_penalty(self.D_aug, x)
            d_loss = d_loss + 0.5 * self.r1_gamma * r1 * self.r1_every  # scaled up: applied lazily
            logs["r1"] = r1.item()
        self.opts["D"].zero_grad(set_to_none=True)
        d_loss.backward()
        self.opts["D"].step()

        # --- G: non-saturating ------------------------------------------------------
        g_loss = F.softplus(-self.D_aug(self._fake(len(x)))).mean()
        self.opts["G"].zero_grad(set_to_none=True)
        g_loss.backward()
        self.opts["G"].step()

        logs.update(d_loss=d_loss.item(), g_loss=g_loss.item(),
                    d_real=d_real.mean().item(), d_fake=d_fake.mean().item())
        return logs

    @torch.no_grad()
    def sample(self, n, truncation=1.0, **kwargs):
        return self.net_for_sampling("G").generate(n, truncation=truncation, device=self.device)

    def export_model(self):
        return self.emas["G"].model
