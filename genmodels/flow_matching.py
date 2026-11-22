"""Flow matching (Lipman et al., 2022) / rectified flow (Liu et al., 2022).

Instead of a noising process, define a straight path between a noise sample
x_0 ~ N(0, I) and a data sample x_1:

    x_t = (1 - (1 - sigma_min) t) x_0 + t x_1,        t in [0, 1]

Its velocity is constant along the path, u = x_1 - (1 - sigma_min) x_0. A network
v_theta(x_t, t) is trained to regress it:

    L = E_{t, x_0, x_1} || v_theta(x_t, t) - u ||^2

The *marginal* field learnt this way transports the whole noise distribution
onto the data distribution, so sampling is solving the ODE dx/dt = v_theta(x, t)
from t=0 (noise) to t=1 (data) with a handful of Euler or Heun steps.

Convention here: t=0 is noise, t=1 is data (the opposite of diffusion's t).
``sigma_min = 0`` is exactly rectified flow.

``time_sampling: logit_normal`` (Esser et al., 2024 / SD3) samples more t near
the middle of the path, where the regression target is hardest.
Classifier-free guidance works as in diffusion, applied to the velocity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .common.trainer import BaseTrainer
from .common.unet import UNet


class FlowMatchingModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_channels: int = 3, image_size: int = 32, num_classes: int = 0, sigma_min: float = 0.0,
                 base_channels: int = 64, channel_mults: list[int] = (1, 2, 2), num_res_blocks: int = 2,
                 attn_resolutions: list[int] = (16,), dropout: float = 0.1):
        super().__init__()
        self.in_channels, self.image_size, self.num_classes, self.sigma_min = in_channels, image_size, num_classes, sigma_min
        self.unet = UNet(in_channels, image_size, base_channels, tuple(channel_mults), num_res_blocks,
                         tuple(attn_resolutions), dropout, num_classes)

    # ---- training ------------------------------------------------------------------
    def sample_t(self, n, device, kind="uniform"):
        if kind == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device))
        return torch.rand(n, device=device)

    def loss(self, x1, y=None, p_uncond: float = 0.0, time_sampling: str = "uniform"):
        x0 = torch.randn_like(x1)
        t = self.sample_t(len(x1), x1.device, time_sampling)
        tb = t[:, None, None, None]
        x_t = (1 - (1 - self.sigma_min) * tb) * x0 + tb * x1
        target = x1 - (1 - self.sigma_min) * x0
        if y is not None and p_uncond > 0:
            drop = torch.rand(len(y), device=y.device) < p_uncond
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return F.mse_loss(self.unet(x_t, t * 1000, y), target)

    # ---- sampling ------------------------------------------------------------------
    def velocity(self, x, t: float, y=None, guidance_scale: float = 1.0):
        tt = torch.full((len(x),), t * 1000, device=x.device)
        if y is None or guidance_scale == 1.0:
            return self.unet(x, tt, y)
        null = torch.full_like(y, self.num_classes)
        v_c, v_u = self.unet(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null])).chunk(2)
        return v_u + guidance_scale * (v_c - v_u)

    @torch.no_grad()
    def sample(self, n, steps=50, solver="euler", y=None, guidance_scale=1.0, device=None,
               return_trajectory=False, x0=None):
        """Integrate dx/dt = v(x, t) from t=0 to 1. Euler: 1 network call per step;
        Heun (2nd order): 2 calls per step, usually better at equal call count."""
        device = device or next(self.parameters()).device
        x = x0 if x0 is not None else torch.randn(n, self.in_channels, self.image_size, self.image_size, device=device)
        ts = torch.linspace(0, 1, steps + 1).tolist()
        traj = [x.cpu()] if return_trajectory else None
        for t, t_next in zip(ts[:-1], ts[1:]):
            dt = t_next - t
            v = self.velocity(x, t, y, guidance_scale)
            if solver == "heun" and t_next < 1:
                x_pred = x + dt * v
                v = 0.5 * (v + self.velocity(x_pred, t_next, y, guidance_scale))
            x = x + dt * v
            if return_trajectory:
                traj.append(x.cpu())
        x = x.clamp(-1, 1)
        return (x, traj) if return_trajectory else x

    @torch.no_grad()
    def generate(self, n: int, steps: int = 50, solver: str = "euler", y=None, guidance_scale: float = 1.0,
                 device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if self.num_classes and y is None:
            y = torch.arange(n, device=device) % self.num_classes
        return self.sample(n, steps, solver, y, guidance_scale, device)


class Trainer(BaseTrainer):
    family = "flow_matching"

    def build(self):
        self.conditional = bool(self.method_cfg.get("conditional", False))
        self.model = FlowMatchingModel(
            in_channels=self.channels, image_size=self.image_size,
            num_classes=self.num_classes if self.conditional else 0,
            sigma_min=float(self.method_cfg.get("sigma_min", 0.0)),
            **self.model_cfg,
        ).to(self.device)
        self.nets["model"] = self.model
        self.base_lr = float(self.train_cfg.get("lr", 2e-4))
        self.opts["model"] = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr,
                                               weight_decay=float(self.train_cfg.get("weight_decay", 0.0)))
        self.add_ema("model")
        self.p_uncond = float(self.method_cfg.get("p_uncond", 0.1)) if self.conditional else 0.0
        self.time_sampling = self.method_cfg.get("time_sampling", "uniform")
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.warmup = int(self.train_cfg.get("warmup_steps", 1000))
        self.sample_steps = int(self.method_cfg.get("sample_steps", 50))
        self.solver = self.method_cfg.get("solver", "euler")
        self.guidance = float(self.method_cfg.get("guidance_scale", 1.0))

    def training_step(self, x, y):
        for g in self.opts["model"].param_groups:
            g["lr"] = self.base_lr * min(1.0, (self.step + 1) / max(1, self.warmup))
        loss = self.model.loss(x, y if self.conditional else None, self.p_uncond, self.time_sampling)
        opt = self.opts["model"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        opt.step()
        return {"loss": loss.item(), "grad_norm": gn.item()}

    @torch.no_grad()
    def sample(self, n, steps=None, solver=None, guidance_scale=None, **kwargs):
        return self.net_for_sampling("model").generate(
            n, steps=steps or self.sample_steps, solver=solver or self.solver,
            guidance_scale=self.guidance if guidance_scale is None else guidance_scale, device=self.device)

    def export_model(self):
        return self.emas["model"].model
