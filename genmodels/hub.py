"""Upload a finished run to the Hugging Face Hub.

A run directory (written by ``BaseTrainer.fit``) contains::

    model/            config.json + model.safetensors  (from PyTorchModelHubMixin)
    config.yaml       the experiment config
    samples.png       fixed-seed sample grid at the last step
    metrics.json      final losses / metrics
    run_info.json     family, steps, hardware, git commit, ...
    tensorboard/      event files

``push_run`` turns that into a model repo: the weights at the repo root (so
``ModelClass.from_pretrained(repo_id)`` works), a generated model card, the
samples and metrics, and the TensorBoard logs (the Hub shows them in a
"Training metrics" tab).

Authentication: run ``huggingface-cli login`` (or ``notebook_login()`` in a
notebook), or set ``HF_TOKEN``. The token is never written anywhere by this code.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

GITHUB_URL = "https://github.com/yar78/vision-generative-models-survey"

_HUB_DATASETS = {
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "cifar10": "uoft-cs/cifar10",
}

_DESCRIPTIONS = {
    "vae": "A convolutional variational autoencoder trained on the ELBO (reconstruction + beta * KL).",
    "gan": "A DCGAN-style generative adversarial network.",
    "stylegan": "A compact StyleGAN2-style generator (mapping network, modulated convolutions, noise inputs) trained with the non-saturating loss and R1 regularisation.",
    "diffusion": "A DDPM: a U-Net trained to predict the noise added by a fixed Gaussian diffusion process; samples with DDPM or DDIM.",
    "flow_matching": "A conditional flow-matching / rectified-flow model: a U-Net velocity field integrated from noise to data with an ODE solver.",
}


def resolve_repo_id(cfg: dict, repo_id: str | None = None) -> str:
    """``repo_id`` if given, else ``hub.repo_id`` from the config, else ``<user>/<name>``
    where ``<user>`` is ``hub.username``, ``$HF_USERNAME`` or the logged-in account."""
    if repo_id:
        return repo_id
    hub = cfg.get("hub") or {}
    if hub.get("repo_id"):
        return hub["repo_id"]
    user = hub.get("username") or os.environ.get("HF_USERNAME")
    if not user:
        from huggingface_hub import whoami

        user = whoami()["name"]
    return f"{user}/{cfg['name']}"


def render_model_card(run_dir: str | os.PathLike, repo_id: str) -> str:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}

    family = info["family"]
    module, cls = info["model_class"].rsplit(".", 1)
    dataset = info.get("dataset")

    front = {
        "library_name": "pytorch",
        "tags": ["genmodels", family.replace("_", "-"), "image-generation", "pytorch_model_hub_mixin"],
        "pipeline_tag": "unconditional-image-generation",
    }
    if dataset in _HUB_DATASETS:
        front["datasets"] = [_HUB_DATASETS[dataset]]
    elif dataset and "/" in dataset:  # a Hub dataset repo, e.g. yaryar78/Ray_dog
        front["datasets"] = [dataset]

    params = info.get("params", {})
    param_str = ", ".join(f"{k}: {v / 1e6:.2f}M" for k, v in params.items())
    commit = info.get("git_commit")
    commit_link = f"[`{commit[:10]}`]({GITHUB_URL}/commit/{commit.removesuffix('-dirty')})" if commit else "unknown"
    if commit and commit.endswith("-dirty"):
        commit_link += " (with uncommitted changes)"

    rows = [
        ("Family", family),
        ("Model class", f"`{info['model_class']}`"),
        ("Dataset", f"[{dataset}](https://huggingface.co/datasets/{dataset})" if dataset and "/" in dataset else dataset),
        ("Resolution", f"{info.get('image_size')}x{info.get('image_size')}"),
        ("Training steps", info.get("steps")),
        ("Batch size", info.get("batch_size")),
        ("Parameters", param_str),
        ("Hardware", info.get("device")),
        ("Training time", f"{info.get('train_hours')} h"),
        ("Code", commit_link),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    metric_table = "\n".join(f"| {k} | {v:.5g} |" for k, v in metrics.items()) or "| - | - |"

    return f"""---
{yaml.safe_dump(front, sort_keys=False).strip()}
---

# {repo_id.split('/')[-1]}

{_DESCRIPTIONS.get(family, '')}

Trained with [vision-generative-models-survey]({GITHUB_URL}).

![samples](samples.png)

## Training

| | |
|---|---|
{table}

The exact config is in [`train_config.yaml`](train_config.yaml); TensorBoard logs are under `runs/`.

## Metrics

| metric | value |
|---|---|
{metric_table}

## Usage

```bash
pip install git+{GITHUB_URL}
```

```python
from {module} import {cls}

model = {cls}.from_pretrained("{repo_id}")
images = model.generate(16)   # tensor in [-1, 1], shape (16, C, H, W)
```
"""


def push_run(
    run_dir: str | os.PathLike,
    repo_id: str | None = None,
    private: bool | None = None,
    include_logs: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload a finished run directory to the Hub and return the repo URL.

    Repos are created private unless ``private=False`` or ``hub.private: false`` is set.
    Pushing the same run again updates the repo with a new commit.
    """
    from huggingface_hub import HfApi

    run_dir = Path(run_dir)
    if not (run_dir / "model").is_dir():
        raise FileNotFoundError(f"{run_dir}/model not found -- has training finished (trainer.export())?")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    repo_id = resolve_repo_id(cfg, repo_id)
    if private is None:
        private = (cfg.get("hub") or {}).get("private", True)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in (run_dir / "model").iterdir():
            shutil.copy2(f, stage / f.name)
        shutil.copy2(run_dir / "config.yaml", stage / "train_config.yaml")
        for name in ("samples.png", "metrics.json", "run_info.json"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, stage / name)
        if include_logs and (run_dir / "tensorboard").is_dir():
            shutil.copytree(run_dir / "tensorboard", stage / "runs")
        (stage / "README.md").write_text(render_model_card(run_dir, repo_id))

        info = json.loads((run_dir / "run_info.json").read_text())
        api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload {info['name']} at step {info['steps']}",
        )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed {run_dir} -> {url} ({'private' if private else 'public'})")
    return url
