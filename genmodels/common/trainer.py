"""The training loop every family shares.

A family subclasses :class:`BaseTrainer` and fills in four methods:

* ``build()``           create networks in ``self.nets``, optimizers in ``self.opts``
                        and (optionally) EMA copies in ``self.emas``
* ``training_step(x,y)`` one optimisation step; returns a dict of floats to log
* ``sample(n)``         return ``n`` images in [-1, 1]
* ``export_model()``    the ``nn.Module`` that gets saved / pushed to the Hub

Everything else -- step counting, TensorBoard, sample grids, checkpoints, resume,
and the final export -- lives here so it behaves the same for every family.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .config import default_output_root, save_config, section
from .data import dataset_info, get_dataloader, infinite
from .utils import EMA, count_params, get_device, save_grid, seed_everything


class BaseTrainer:
    family: str = ""

    def __init__(self, cfg: dict, run_dir=None, device=None):
        self.cfg = cfg
        self.data_cfg = section(cfg, "data")
        self.model_cfg = section(cfg, "model")
        self.method_cfg = section(cfg, "method")
        self.train_cfg = section(cfg, "train")

        self.device = get_device(device or self.train_cfg.get("device"))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        seed_everything(cfg.get("seed", 0))

        self.channels, self.num_classes = dataset_info(self.data_cfg)
        self.image_size = self.data_cfg.get("image_size", 32)
        self.run_dir = Path(run_dir) if run_dir else default_output_root() / cfg["name"]

        self.step = 0
        self.nets: dict[str, nn.Module] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        self.emas: dict[str, EMA] = {}
        self.history: list[dict] = []
        self._train_seconds = 0.0
        self.build()

    # ---- to implement per family -------------------------------------------------
    def build(self) -> None:
        raise NotImplementedError

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def export_model(self) -> nn.Module:
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------------------
    def add_ema(self, name: str) -> None:
        decay = float(self.train_cfg.get("ema_decay", 0.999))
        self.emas[name] = EMA(self.nets[name], decay)

    def net_for_sampling(self, name: str) -> nn.Module:
        """The EMA copy if there is one, otherwise the live network."""
        return self.emas[name].model if name in self.emas else self.nets[name]

    @torch.no_grad()
    def sample_fixed(self, n: int, seed: int = 1234, **kwargs) -> torch.Tensor:
        """Sample with a fixed seed, so grids from different steps are comparable."""
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return self.sample(n, **kwargs)

    def num_params(self) -> dict[str, int]:
        return {k: count_params(v) for k, v in self.nets.items()}

    # ---- checkpoints -----------------------------------------------------------------
    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    def save_checkpoint(self) -> None:
        state = {
            "step": self.step,
            "train_seconds": self._train_seconds,
            "nets": {k: v.state_dict() for k, v in self.nets.items()},
            "opts": {k: v.state_dict() for k, v in self.opts.items()},
            "emas": {k: v.state_dict() for k, v in self.emas.items()},
            "history": self.history,
        }
        tmp = self.ckpt_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)  # never leave a half-written checkpoint

    def load_checkpoint(self, path=None) -> None:
        state = torch.load(path or self.ckpt_path, map_location=self.device, weights_only=False)
        for k, v in state["nets"].items():
            self.nets[k].load_state_dict(v)
        for k, v in state["opts"].items():
            self.opts[k].load_state_dict(v)
        for k, v in state["emas"].items():
            self.emas[k].load_state_dict(v)
        self.step = state["step"]
        self._train_seconds = state.get("train_seconds", 0.0)
        self.history = state.get("history", [])

    # ---- the loop ----------------------------------------------------------------------
    def fit(self, steps: int | None = None, progress: bool = True) -> list[dict]:
        """Train until ``steps`` total optimisation steps (default: ``train.steps``).

        Calling ``fit`` again continues from where it stopped. If the run directory
        already holds a checkpoint and ``train.resume`` is true (the default), training
        resumes from it -- which is what you want after a pod gets terminated.
        """
        from torch.utils.tensorboard import SummaryWriter

        total = int(steps or self.train_cfg.get("steps", 10_000))
        log_every = int(self.train_cfg.get("log_every", 100))
        sample_every = int(self.train_cfg.get("sample_every", 1000))
        ckpt_every = int(self.train_cfg.get("ckpt_every", 5000))
        n_samples = int(self.train_cfg.get("n_samples", 64))

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        save_config(self.cfg, self.run_dir / "config.yaml")

        if self.step == 0 and self.train_cfg.get("resume", True) and self.ckpt_path.exists():
            self.load_checkpoint()
            print(f"Resumed from {self.ckpt_path} at step {self.step}")
        if self.step >= total:
            print(f"Already trained for {self.step} >= {total} steps; nothing to do.")
            return self.history

        writer = SummaryWriter(str(self.run_dir / "tensorboard"), purge_step=self.step)
        loader = infinite(get_dataloader(self.data_cfg, train=True))
        running = defaultdict(list)
        for net in self.nets.values():
            net.train()

        bar = tqdm(total=total, initial=self.step, disable=not progress, desc=self.cfg["name"])
        t0 = time.time()
        try:
            while self.step < total:
                x, y = next(loader)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logs = self.training_step(x, y)
                for name, ema in self.emas.items():
                    ema.update(self.nets[name])
                self.step += 1
                bar.update(1)
                for k, v in logs.items():
                    running[k].append(float(v))

                last = self.step == total
                if self.step % log_every == 0 or last:
                    means = {k: sum(v) / len(v) for k, v in running.items()}
                    for k, v in means.items():
                        writer.add_scalar(f"train/{k}", v, self.step)
                    self.history.append({"step": self.step, **means})
                    bar.set_postfix({k: f"{v:.4f}" for k, v in means.items()})
                    running.clear()
                if self.step % sample_every == 0 or last:
                    self._log_samples(writer, n_samples)
                if self.step % ckpt_every == 0 or last:
                    self._train_seconds += time.time() - t0
                    t0 = time.time()
                    self.save_checkpoint()
        finally:
            bar.close()
            writer.close()

        self.export()
        return self.history

    def _log_samples(self, writer, n: int) -> None:
        for net in self.nets.values():
            net.eval()
        imgs = self.sample_fixed(n)
        for net in self.nets.values():
            net.train()
        save_grid(imgs, self.run_dir / "samples" / f"step_{self.step:07d}.png")
        save_grid(imgs, self.run_dir / "samples.png")
        from .utils import image_grid

        writer.add_image("samples", image_grid(imgs), self.step)

    # ---- export ------------------------------------------------------------------------
    def export(self) -> Path:
        """Write everything the Hub upload needs into the run directory:
        ``model/`` (safetensors + config.json), ``metrics.json`` and ``run_info.json``."""
        model = self.export_model()
        model.save_pretrained(self.run_dir / "model")

        # Final losses: average over the last 10% of logged points.
        metrics = {}
        if self.history:
            tail = self.history[-max(1, len(self.history) // 10):]
            keys = [k for k in tail[-1] if k != "step"]
            metrics = {f"final_{k}": sum(h[k] for h in tail if k in h) / len(tail) for k in keys}
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        info = {
            "family": self.family,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "name": self.cfg["name"],
            "dataset": self.data_cfg.get("hf_repo") or self.data_cfg.get("dataset"),
            "image_size": self.image_size,
            "batch_size": self.data_cfg.get("batch_size", 128),
            "steps": self.step,
            "train_hours": round(self._train_seconds / 3600, 3),
            "params": self.num_params(),
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else platform.processor() or self.device.type,
            "git_commit": _git_commit(),
        }
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)
        return self.run_dir


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None
