"""Variational autoencoder (Kingma & Welling, 2013) and beta-VAE (Higgins et al., 2017).

The encoder maps an image x to a Gaussian q(z|x) = N(mu, diag(sigma^2)); the decoder
maps a latent z back to an image. Training maximises the ELBO

    log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || N(0, I))

which, with a Gaussian decoder, is a reconstruction (MSE) term plus a KL term.
``beta`` > 1 weights the KL more heavily (beta-VAE): blurrier samples, more
disentangled latents.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .common.trainer import BaseTrainer


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


class VAE(nn.Module, PyTorchModelHubMixin):
    """Convolutional VAE. ``image_size`` must be divisible by 2**len(channel_mults)."""

    def __init__(self, in_channels: int = 1, image_size: int = 32, latent_dim: int = 32,
                 base_channels: int = 32, channel_mults: list[int] = (1, 2, 4)):
        super().__init__()
        self.latent_dim = latent_dim
        chans = [base_channels * m for m in channel_mults]
        self.bottom = image_size // 2 ** len(chans)
        flat = chans[-1] * self.bottom ** 2

        enc, c = [], in_channels
        for ch in chans:
            enc.append(_down(c, ch))
            c = ch
        self.encoder = nn.Sequential(*enc, nn.Flatten())
        self.to_mu = nn.Linear(flat, latent_dim)
        self.to_logvar = nn.Linear(flat, latent_dim)

        self.from_z = nn.Linear(latent_dim, flat)
        dec = []
        for ch in reversed(chans[:-1]):
            dec.append(_up(c, ch))
            c = ch
        self.decoder = nn.Sequential(*dec, nn.ConvTranspose2d(c, in_channels, 4, 2, 1), nn.Tanh())
        self._top = chans[-1]

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        h = self.from_z(z).view(-1, self._top, self.bottom, self.bottom)
        return self.decoder(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps: sampling made differentiable w.r.t. mu and sigma."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def generate(self, n: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0):
    """Negative ELBO per image: sum-of-squares reconstruction + beta * KL to N(0, I).

    KL(N(mu, sigma^2) || N(0, 1)) has the closed form
    0.5 * sum(mu^2 + sigma^2 - log sigma^2 - 1).
    """
    recon = F.mse_loss(x_hat, x, reduction="sum") / len(x)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / len(x)
    return recon + beta * kl, recon, kl


class Trainer(BaseTrainer):
    family = "vae"

    def build(self):
        self.model = VAE(in_channels=self.channels, image_size=self.image_size, **self.model_cfg).to(self.device)
        self.nets["vae"] = self.model
        self.opts["vae"] = torch.optim.Adam(self.model.parameters(), lr=float(self.train_cfg.get("lr", 1e-3)))
        self.beta = float(self.method_cfg.get("beta", 1.0))
        # KL warm-up: ramp beta from 0 over the first N steps to avoid posterior collapse.
        self.kl_warmup = int(self.method_cfg.get("kl_warmup_steps", 0))

    def training_step(self, x, y):
        beta = self.beta * min(1.0, self.step / self.kl_warmup) if self.kl_warmup else self.beta
        x_hat, mu, logvar = self.model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta)
        opt = self.opts["vae"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return {"loss": loss.item(), "recon": recon.item(), "kl": kl.item(), "beta": beta}

    @torch.no_grad()
    def sample(self, n, **kwargs):
        return self.model.generate(n, self.device)

    @torch.no_grad()
    def reconstruct(self, x):
        self.model.eval()
        mu, _ = self.model.encode(x.to(self.device))
        return self.model.decode(mu)

    def export_model(self):
        return self.model
