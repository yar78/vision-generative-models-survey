"""A few optimisation steps of every config on fake data, on CPU: shapes, losses,
checkpoint/resume and the Hub save -> from_pretrained round trip (local, no network)."""

import importlib
import json
from pathlib import Path

import pytest
import torch

from genmodels import build_trainer
from genmodels.common.config import load_config
from genmodels.hub import render_model_card

CONFIGS = sorted(str(p) for p in (Path(__file__).parents[1] / "configs").rglob("*.yaml"))


def tiny(cfg):
    cfg["data"].update(dataset="fake", batch_size=4, num_workers=0, fake_size=16,
                       channels=1 if "mnist" in cfg["data"]["dataset"] else 3)
    cfg["train"].update(steps=2, log_every=1, sample_every=2, ckpt_every=2, n_samples=4, device="cpu")
    m = cfg.get("model", {})
    for k in ("base_channels",):
        if k in m:
            m[k] = 16
    for k in ("channels", "max_channels", "d_channels", "z_dim", "w_dim"):
        if k in m:
            m[k] = 32
    for k in ("g_channels",):
        if k in m:
            m[k] = 16
    if "timesteps" in cfg.get("method", {}):
        cfg["method"]["timesteps"] = 20
        cfg["method"]["sample_steps"] = 3
    if "sample_steps" in cfg.get("method", {}):
        cfg["method"]["sample_steps"] = 3
    return cfg


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: "/".join(Path(p).parts[-2:]))
def test_train_export_reload(config, tmp_path):
    cfg = tiny(load_config(config))
    trainer = build_trainer(cfg, run_dir=tmp_path)
    history = trainer.fit(progress=False)
    assert history and all(torch.isfinite(torch.tensor(v)) for h in history for v in h.values())

    for f in ("model/config.json", "model/model.safetensors", "samples.png", "metrics.json", "run_info.json", "checkpoint.pt"):
        assert (tmp_path / f).exists(), f

    # Hub round trip, locally: the exported class reloads from the saved folder and samples.
    info = json.loads((tmp_path / "run_info.json").read_text())
    module, cls = info["model_class"].rsplit(".", 1)
    Model = getattr(importlib.import_module(module), cls)
    model = Model.from_pretrained(tmp_path / "model")
    gen_kwargs = {"steps": 2} if cfg["family"] in ("diffusion", "flow_matching") else {}
    x = model.generate(2, **gen_kwargs)
    size = cfg["data"]["image_size"]
    assert x.shape == (2, cfg["data"]["channels"], size, size)
    assert x.min() >= -1 and x.max() <= 1

    card = render_model_card(tmp_path, "someone/" + cfg["name"])
    assert f"from {module} import {cls}" in card

    # Resume continues from the checkpoint instead of restarting.
    cfg["train"]["steps"] = 3
    again = build_trainer(cfg, run_dir=tmp_path)
    again.fit(progress=False)
    assert again.step == 3


def test_diffaugment_keeps_shape_and_gradients():
    from genmodels.common.diffaugment import diff_augment

    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    y = diff_augment(x, "color,translation,cutout")
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
