"""Generative adversarial networks: DCGAN (Radford et al., 2015) with two losses.

A generator G maps noise z ~ N(0, I) to an image; a discriminator D scores images
as real or fake. They play the minimax game

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

``loss: bce``      the original game with the *non-saturating* generator loss
                   -log D(G(z)), which gives G useful gradients early on.
``loss: wgan-gp``  Wasserstein GAN with gradient penalty (Gulrajani et al., 2017):
                   D becomes a critic estimating the Wasserstein-1 distance, kept
                   1-Lipschitz by penalising ||grad D|| != 1 on interpolates.

``num_classes > 0`` makes both networks class-conditional (a cGAN): the label is
embedded and fed to G with z, and to D as extra input channels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .common.diffaugment import diff_augment
from .common.trainer import BaseTrainer


class Generator(nn.Module, PyTorchModelHubMixin):
    """DCGAN generator: z -> 4x4 feature map -> transposed convs doubling resolution -> tanh."""

    def __init__(self, out_channels: int = 3, image_size: int = 32, z_dim: int = 128,
                 base_channels: int = 64, num_classes: int = 0):
        super().__init__()
        self.z_dim, self.num_classes = z_dim, num_classes
        n_up = int(math.log2(image_size)) - 2  # 4 -> image_size
        ch = base_channels * 2 ** (n_up - 1)
        in_dim = z_dim + (z_dim if num_classes else 0)
        self.embed = nn.Embedding(num_classes, z_dim) if num_classes else None
        self.project = nn.Sequential(nn.ConvTranspose2d(in_dim, ch, 4, 1, 0), nn.BatchNorm2d(ch), nn.ReLU(True))
        blocks = []
        for _ in range(n_up - 1):
            blocks += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.BatchNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        blocks += [nn.ConvTranspose2d(ch, out_channels, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*blocks)

    def forward(self, z, y=None):
        if self.embed is not None:
            z = torch.cat([z, self.embed(y)], dim=1)
        return self.net(self.project(z[:, :, None, None]))

    @torch.no_grad()
    def generate(self, n: int, y=None, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        if self.num_classes and y is None:
            y = torch.arange(n, device=device) % self.num_classes
        return self(z, y)


class Discriminator(nn.Module):
    """DCGAN discriminator. ``norm='batch'`` for BCE; WGAN-GP needs no batch norm
    (the penalty is per-sample), so it uses layer-style group norm instead."""

    def __init__(self, in_channels: int = 3, image_size: int = 32, base_channels: int = 64,
                 num_classes: int = 0, norm: str = "batch"):
        super().__init__()
        self.image_size = image_size
        self.embed = nn.Embedding(num_classes, image_size * image_size) if num_classes else None
        n_down = int(math.log2(image_size)) - 2
        c, ch = in_channels + (1 if num_classes else 0), base_channels
        layers = [nn.Conv2d(c, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        for _ in range(n_down - 1):
            norm_layer = nn.BatchNorm2d(ch * 2) if norm == "batch" else nn.GroupNorm(1, ch * 2)
            layers += [nn.Conv2d(ch, ch * 2, 4, 2, 1), norm_layer, nn.LeakyReLU(0.2, True)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 4, 1, 0)]  # 4x4 -> 1x1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x, y=None):
        if self.embed is not None:
            x = torch.cat([x, self.embed(y).view(-1, 1, self.image_size, self.image_size)], dim=1)
        return self.net(x).view(-1)


def init_dcgan_weights(m):
    """DCGAN paper init: N(0, 0.02) for convs, N(1, 0.02) for norm scales."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def d_loss_bce(d_real, d_fake):
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()  # -log D(x) - log(1 - D(G(z)))


def g_loss_bce(d_fake):
    return F.softplus(-d_fake).mean()  # non-saturating: -log D(G(z))


def gradient_penalty(D, real, fake, y=None):
    """WGAN-GP: E[(||grad_x D(x_hat)|| - 1)^2] on random interpolates x_hat."""
    eps = torch.rand(len(real), 1, 1, 1, device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(D(x_hat, y).sum(), x_hat, create_graph=True)
    return (grad.flatten(1).norm(dim=1) - 1).pow(2).mean()


class Trainer(BaseTrainer):
    family = "gan"

    def build(self):
        m = dict(self.model_cfg)
        self.loss_type = self.method_cfg.get("loss", "bce")
        self.conditional = bool(self.method_cfg.get("conditional", False))
        n_cls = self.num_classes if self.conditional else 0
        z_dim = m.get("z_dim", 128)
        g_ch, d_ch = m.get("g_channels", 64), m.get("d_channels", 64)

        self.G = Generator(self.channels, self.image_size, z_dim, g_ch, n_cls).to(self.device)
        norm = "batch" if self.loss_type == "bce" else "group"
        self.D = Discriminator(self.channels, self.image_size, d_ch, n_cls, norm).to(self.device)
        self.G.apply(init_dcgan_weights)
        self.D.apply(init_dcgan_weights)
        self.nets = {"G": self.G, "D": self.D}

        lr = float(self.train_cfg.get("lr", 2e-4))
        betas = (0.5, 0.999) if self.loss_type == "bce" else (0.0, 0.9)
        self.opts = {"G": torch.optim.Adam(self.G.parameters(), lr, betas=betas),
                     "D": torch.optim.Adam(self.D.parameters(), lr, betas=betas)}
        self.n_critic = int(self.method_cfg.get("n_critic", 1 if self.loss_type == "bce" else 5))
        self.gp_weight = float(self.method_cfg.get("gp_weight", 10.0))
        self.augment = self.method_cfg.get("diffaugment")  # e.g. "color,translation,cutout" for small datasets

    def D_aug(self, x, y=None):
        """D sees real and fake images through the same random differentiable augmentation."""
        return self.D(diff_augment(x, self.augment), y)

    def training_step(self, x, y):
        y = y if self.conditional else None
        z_dim = self.G.z_dim
        logs = {}

        # --- discriminator / critic --------------------------------------------------
        for _ in range(self.n_critic):
            z = torch.randn(len(x), z_dim, device=self.device)
            with torch.no_grad():
                fake = self.G(z, y)
            d_real, d_fake = self.D_aug(x, y), self.D_aug(fake, y)
            if self.loss_type == "bce":
                d_loss = d_loss_bce(d_real, d_fake)
            else:
                gp = gradient_penalty(self.D, x, fake, y)
                d_loss = d_fake.mean() - d_real.mean() + self.gp_weight * gp
                logs["gp"] = gp.item()
                logs["wasserstein"] = (d_real.mean() - d_fake.mean()).item()
            self.opts["D"].zero_grad(set_to_none=True)
            d_loss.backward()
            self.opts["D"].step()

        # --- generator ----------------------------------------------------------------
        z = torch.randn(len(x), z_dim, device=self.device)
        d_fake = self.D_aug(self.G(z, y), y)
        g_loss = g_loss_bce(d_fake) if self.loss_type == "bce" else -d_fake.mean()
        self.opts["G"].zero_grad(set_to_none=True)
        g_loss.backward()
        self.opts["G"].step()

        logs.update(d_loss=d_loss.item(), g_loss=g_loss.item())
        if self.loss_type == "bce":
            logs.update(d_real=torch.sigmoid(d_real).mean().item(), d_fake=torch.sigmoid(d_fake).mean().item())
        return logs

    @torch.no_grad()
    def sample(self, n, y=None, **kwargs):
        return self.G.generate(n, y=y, device=self.device)

    def export_model(self):
        return self.G
