"""Experiment configs: plain nested dicts loaded from YAML, with ``a.b=value`` overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    """Apply one ``section.key=value`` override in place. The value is parsed as YAML,
    so ``train.steps=500``, ``train.lr=2e-4`` and ``hub.private=false`` all get the right type."""
    if "=" not in item:
        raise ValueError(f"Override must look like key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def section(cfg: dict, name: str) -> dict:
    return copy.deepcopy(cfg.get(name) or {})


def default_output_root() -> Path:
    """Where runs go: $GENMODELS_OUTPUT, else the RunPod network volume, else ./outputs."""
    if os.environ.get("GENMODELS_OUTPUT"):
        return Path(os.environ["GENMODELS_OUTPUT"])
    if Path("/workspace").is_dir():
        return Path("/workspace/genmodels/outputs")
    return Path("outputs")


def default_data_root() -> Path:
    """Where datasets are downloaded: $GENMODELS_DATA, else the network volume, else ./data."""
    if os.environ.get("GENMODELS_DATA"):
        return Path(os.environ["GENMODELS_DATA"])
    if Path("/workspace").is_dir():
        return Path("/workspace/genmodels/data")
    return Path("data")
