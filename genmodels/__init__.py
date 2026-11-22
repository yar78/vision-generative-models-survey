"""genmodels: small, readable implementations of the main image generative model families.

Each family lives in its own module and exposes the same three things:

* a model class that inherits ``PyTorchModelHubMixin`` (so ``.push_to_hub`` /
  ``.from_pretrained`` work out of the box),
* a ``Trainer`` subclass of :class:`genmodels.common.trainer.BaseTrainer`,
* family-specific sampling helpers.

``FAMILIES`` maps the ``family:`` key of a config file to its module.
"""

import importlib

FAMILIES = {
    "vae": "genmodels.vae",
    "gan": "genmodels.gan",
    "stylegan": "genmodels.stylegan",
    "diffusion": "genmodels.diffusion",
    "flow_matching": "genmodels.flow_matching",
    "dreambooth_lora": "genmodels.dreambooth_lora",  # needs the [finetune] extras
}


def get_family(name: str):
    """Import and return the module implementing a model family."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown family {name!r}; choose one of {sorted(FAMILIES)}")
    return importlib.import_module(FAMILIES[name])


def build_trainer(cfg: dict, run_dir=None, device=None):
    """Build the right Trainer for ``cfg['family']``."""
    return get_family(cfg["family"]).Trainer(cfg, run_dir=run_dir, device=device)
