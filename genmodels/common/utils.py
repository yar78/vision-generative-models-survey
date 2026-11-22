"""Small helpers shared by every family: seeding, device, EMA, image grids."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid, save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class EMA:
    """Exponential moving average of a model's weights.

    Sampling from the averaged weights gives visibly cleaner images for diffusion,
    flow matching and StyleGAN, at no training cost.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], clamped."""
    return (x.clamp(-1, 1) + 1) / 2


def image_grid(x: torch.Tensor, nrow: int | None = None) -> torch.Tensor:
    nrow = nrow or int(len(x) ** 0.5)
    return make_grid(to_unit(x.detach().float().cpu()), nrow=nrow, padding=2)


def save_grid(x: torch.Tensor, path, nrow: int | None = None) -> None:
    save_image(image_grid(x, nrow), path)


def show_grid(x: torch.Tensor, nrow: int | None = None, title: str | None = None, size: float = 6):
    """Display a batch of [-1, 1] images inline (notebooks)."""
    import matplotlib.pyplot as plt

    grid = image_grid(x, nrow).permute(1, 2, 0).numpy()
    plt.figure(figsize=(size, size))
    plt.imshow(grid.squeeze(), cmap="gray" if grid.shape[-1] == 1 else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def plot_history(history: list[dict], keys: list[str] | None = None, title: str | None = None):
    """Plot logged training curves (``trainer.history``) inline."""
    import matplotlib.pyplot as plt

    if not history:
        print("No history yet -- train first.")
        return
    keys = keys or [k for k in history[-1] if k != "step"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), squeeze=False)
    steps = [h["step"] for h in history]
    for ax, k in zip(axes[0], keys):
        ax.plot(steps, [h.get(k, float("nan")) for h in history])
        ax.set_title(k)
        ax.set_xlabel("step")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_images(dataset, max_images: int = 10_000) -> torch.Tensor:
    """Stack up to ``max_images`` images of a dataset into one tensor."""
    n = min(len(dataset), max_images)
    return torch.stack([dataset[i][0] for i in range(n)])


@torch.no_grad()
def nearest_neighbours(samples: torch.Tensor, train_images: torch.Tensor, k: int = 1) -> torch.Tensor:
    """For each sample, the ``k`` closest training images by pixel L2 distance.

    The standard memorisation check: if generated images are near-copies of their
    nearest training image, the model has memorised rather than generalised --
    a real risk with small datasets such as a few hundred photos.
    Returns a tensor of shape (len(samples), k, C, H, W).
    """
    s = samples.flatten(1).float().cpu()
    t = train_images.flatten(1).float().cpu()
    d = torch.cdist(s, t)
    idx = d.topk(k, largest=False).indices
    return train_images[idx]
