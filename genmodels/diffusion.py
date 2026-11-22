"""Denoising diffusion: DDPM (Ho et al., 2020), DDIM sampling (Song et al., 2020) and
classifier-free guidance (Ho & Salimans, 2022).

Forward process (fixed, no learning): add Gaussian noise over T steps,

    q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I ),   abar_t = prod_{s<=t} (1 - beta_s)

so x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps can be sampled in one shot.

Reverse process (learned): a U-Net eps_theta(x_t, t) predicts the noise; the
simplified DDPM objective is just  E || eps - eps_theta(x_t, t) ||^2.

Sampling walks back from x_T ~ N(0, I):
* ``ddpm``: the ancestral sampler, all T steps, stochastic.
* ``ddim``: a deterministic (eta=0) sampler on a subsequence of timesteps --
  50 steps instead of 1000 with little quality loss. Same trained model.

With ``conditional: true`` the label is dropped to a null class with
probability ``p_uncond`` during training, and at sampling

    eps = eps_uncond + w * (eps_cond - eps_uncond)

with guidance scale w (w=1: plain conditional, w>1: sharper, less diverse).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .common.trainer import BaseTrainer
from .common.unet import UNet


def make_beta_schedule(kind: str, T: int) -> torch.Tensor:
    if kind == "linear":  # Ho et al.
        return torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    if kind == "cosine":  # Nichol & Dhariwal: abar_t follows a squared cosine
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        abar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        return (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    raise ValueError(f"Unknown schedule {kind!r}")


class DiffusionModel(nn.Module, PyTorchModelHubMixin):
    """U-Net + noise schedule. The schedule is rebuilt from the config, not stored."""

    def __init__(self, in_channels: int = 3, image_size: int = 32, num_classes: int = 0,
                 timesteps: int = 1000, schedule: str = "linear",
                 base_channels: int = 64, channel_mults: list[int] = (1, 2, 2), num_res_blocks: int = 2,
                 attn_resolutions: list[int] = (16,), dropout: float = 0.1):
        super().__init__()
        self.in_channels, self.image_size, self.num_classes, self.T = in_channels, image_size, num_classes, timesteps
        self.unet = UNet(in_channels, image_size, base_channels, tuple(channel_mults), num_res_blocks,
                         tuple(attn_resolutions), dropout, num_classes)
        betas = make_beta_schedule(schedule, timesteps)
        abar = torch.cumprod(1 - betas, dim=0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        for name, val in {
            "betas": betas,
            "alphas_cumprod": abar,
            "alphas_cumprod_prev": abar_prev,
            "sqrt_abar": abar.sqrt(),
            "sqrt_one_minus_abar": (1 - abar).sqrt(),
            # posterior q(x_{t-1} | x_t, x_0) variance (Ho et al. eq. 7)
            "posterior_var": betas * (1 - abar_prev) / (1 - abar),
        }.items():
            self.register_buffer(name, val.float(), persistent=False)

    @staticmethod
    def _at(a, t):
        return a[t][:, None, None, None]

    # ---- training ------------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        """Jump straight to step t of the forward process."""
        return self._at(self.sqrt_abar, t) * x0 + self._at(self.sqrt_one_minus_abar, t) * noise

    def loss(self, x0, y=None, p_uncond: float = 0.0):
        t = torch.randint(0, self.T, (len(x0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if y is not None and p_uncond > 0:
            drop = torch.rand(len(y), device=y.device) < p_uncond
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return F.mse_loss(self.unet(x_t, t, y), noise)

    # ---- sampling ------------------------------------------------------------------
    def predict_eps(self, x, t, y=None, guidance_scale: float = 1.0):
        tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        if y is None or guidance_scale == 1.0:
            return self.unet(x, tt, y)
        null = torch.full_like(y, self.num_classes)
        eps_c, eps_u = self.unet(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null])).chunk(2)
        return eps_u + guidance_scale * (eps_c - eps_u)

    def _start(self, n, device):
        return torch.randn(n, self.in_channels, self.image_size, self.image_size, device=device)

    @torch.no_grad()
    def ddpm_sample(self, n, y=None, guidance_scale=1.0, device=None, return_trajectory=False):
        device = device or self.betas.device
        x, traj = self._start(n, device), []
        for t in reversed(range(self.T)):
            eps = self.predict_eps(x, t, y, guidance_scale)
            abar, beta = self.alphas_cumprod[t], self.betas[t]
            # mean of p(x_{t-1} | x_t) = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - abar_t) * eps)
            mean = (x - beta / (1 - abar).sqrt() * eps) / (1 - beta).sqrt()
            x = mean + self.posterior_var[t].sqrt() * torch.randn_like(x) if t > 0 else mean
            if return_trajectory and t % (self.T // 10) == 0:
                traj.append(x.clamp(-1, 1).cpu())
        return (x.clamp(-1, 1), traj) if return_trajectory else x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, n, steps=50, eta=0.0, y=None, guidance_scale=1.0, device=None, return_trajectory=False):
        device = device or self.betas.device
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        x, traj = self._start(n, device), []
        for i, t in enumerate(ts):
            eps = self.predict_eps(x, t, y, guidance_scale)
            abar = self.alphas_cumprod[t]
            abar_next = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x0 = ((x - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)   # predicted clean image
            eps = (x - abar.sqrt() * x0) / (1 - abar).sqrt()                  # re-derive eps after clamping
            sigma = eta * ((1 - abar_next) / (1 - abar) * (1 - abar / abar_next)).sqrt()
            x = abar_next.sqrt() * x0 + (1 - abar_next - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn_like(x)
            if return_trajectory:
                traj.append(x0.cpu())
        return (x.clamp(-1, 1), traj) if return_trajectory else x.clamp(-1, 1)

    @torch.no_grad()
    def generate(self, n: int, sampler: str = "ddim", steps: int = 50, y=None, guidance_scale: float = 1.0,
                 device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if self.num_classes and y is None:
            y = torch.arange(n, device=device) % self.num_classes
        if sampler == "ddpm":
            return self.ddpm_sample(n, y, guidance_scale, device)
        return self.ddim_sample(n, steps, 0.0, y, guidance_scale, device)


class Trainer(BaseTrainer):
    family = "diffusion"

    def build(self):
        self.conditional = bool(self.method_cfg.get("conditional", False))
        self.model = DiffusionModel(
            in_channels=self.channels, image_size=self.image_size,
            num_classes=self.num_classes if self.conditional else 0,
            timesteps=int(self.method_cfg.get("timesteps", 1000)),
            schedule=self.method_cfg.get("schedule", "linear"),
            **self.model_cfg,
        ).to(self.device)
        self.nets["model"] = self.model
        self.opts["model"] = torch.optim.AdamW(self.model.parameters(), lr=float(self.train_cfg.get("lr", 2e-4)),
                                               weight_decay=float(self.train_cfg.get("weight_decay", 0.0)))
        self.add_ema("model")
        self.p_uncond = float(self.method_cfg.get("p_uncond", 0.1)) if self.conditional else 0.0
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.warmup = int(self.train_cfg.get("warmup_steps", 1000))
        self.base_lr = float(self.train_cfg.get("lr", 2e-4))
        self.sample_steps = int(self.method_cfg.get("sample_steps", 50))
        self.guidance = float(self.method_cfg.get("guidance_scale", 1.0))

    def training_step(self, x, y):
        for g in self.opts["model"].param_groups:  # linear LR warm-up
            g["lr"] = self.base_lr * min(1.0, (self.step + 1) / max(1, self.warmup))
        loss = self.model.loss(x, y if self.conditional else None, self.p_uncond)
        opt = self.opts["model"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        opt.step()
        return {"loss": loss.item(), "grad_norm": gn.item()}

    @torch.no_grad()
    def sample(self, n, sampler="ddim", steps=None, guidance_scale=None, **kwargs):
        return self.net_for_sampling("model").generate(
            n, sampler=sampler, steps=steps or self.sample_steps,
            guidance_scale=self.guidance if guidance_scale is None else guidance_scale, device=self.device)

    def export_model(self):
        return self.emas["model"].model
