"""Datasets. Every image is returned as a float tensor scaled to [-1, 1]."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import default_data_root

# name -> (torchvision class, channels, num_classes)
_TORCHVISION = {
    "mnist": (datasets.MNIST, 1, 10),
    "fashion_mnist": (datasets.FashionMNIST, 1, 10),
    "cifar10": (datasets.CIFAR10, 3, 10),
}


class FakeImages(Dataset):
    """Random images, for smoke tests that must not touch the network."""

    def __init__(self, n=256, channels=3, image_size=32, num_classes=10):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, channels, image_size, image_size, generator=g) * 2 - 1
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class HFImages(Dataset):
    """An image dataset from the Hugging Face Hub, e.g. ``yaryar78/Ray_dog``.

    Photos are decoded once and downscaled so the short side is ``cache_size``
    pixels, then kept in memory: decoding a 2000-px JPEG every step would
    dominate training time for a small dataset. Random crops / flips are applied
    on the fly by ``transform``.
    """

    def __init__(self, repo: str, split: str = "train", image_column: str | None = None,
                 label_column: str | None = None, cache_size: int = 96, transform=None, cache_dir=None):
        from datasets import Image as HFImage
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, cache_dir=str(cache_dir) if cache_dir else None)
        if image_column is None:  # first column whose feature type is Image
            image_column = next((k for k, f in ds.features.items() if isinstance(f, HFImage)), None)
            if image_column is None:
                raise ValueError(f"No image column found in {repo}; set data.image_column")
        self.images, self.labels = [], []
        for ex in ds:
            img = ex[image_column].convert("RGB")
            img = transforms.functional.resize(img, cache_size, antialias=True)
            self.images.append(img)
            self.labels.append(int(ex[label_column]) if label_column else 0)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i]
        return (self.transform(x) if self.transform else x), self.labels[i]


def dataset_info(data_cfg: dict) -> tuple[int, int]:
    """(channels, num_classes) for a data config, without loading anything."""
    name = data_cfg["dataset"]
    if name in _TORCHVISION:
        return _TORCHVISION[name][1], _TORCHVISION[name][2]
    if name == "fake":
        return data_cfg.get("channels", 3), data_cfg.get("num_classes", 10)
    if name in ("folder", "hf"):
        return 3, data_cfg.get("num_classes", 0)
    raise ValueError(f"Unknown dataset {name!r}")


def get_dataset(data_cfg: dict, train: bool = True) -> Dataset:
    """Build a dataset from the ``data:`` section of a config.

    Supported ``dataset`` values: ``mnist``, ``fashion_mnist``, ``cifar10``,
    ``hf`` (a Hugging Face Hub dataset given by ``hf_repo``),
    ``folder`` (any ImageFolder-style directory given by ``path``) and ``fake``.

    Augmentation keys: ``hflip`` (random horizontal flip) and ``random_crop``
    (random square crop covering ``crop_scale`` of the image, instead of a centre
    crop -- useful for small photo datasets with varied framing).
    """
    name = data_cfg["dataset"]
    size = data_cfg.get("image_size", 32)
    channels, num_classes = dataset_info(data_cfg)
    if name == "fake":
        return FakeImages(data_cfg.get("fake_size", 256), channels, size, num_classes)

    if train and data_cfg.get("random_crop", False):
        lo, hi = data_cfg.get("crop_scale", [0.6, 1.0])
        tfs = [transforms.RandomResizedCrop(size, scale=(lo, hi), ratio=(1.0, 1.0), antialias=True)]
    else:
        tfs = [transforms.Resize(size, antialias=True), transforms.CenterCrop(size)]
    if train and data_cfg.get("hflip", False):
        tfs.append(transforms.RandomHorizontalFlip())
    tfs += [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
    tf = transforms.Compose(tfs)

    if name == "hf":
        return HFImages(
            data_cfg["hf_repo"], split=data_cfg.get("hf_split", "train"),
            image_column=data_cfg.get("image_column"), label_column=data_cfg.get("label_column"),
            cache_size=int(data_cfg.get("cache_size", round(size * 1.5))), transform=tf,
            cache_dir=data_cfg.get("root") or default_data_root() / "hf",
        )
    if name == "folder":
        return datasets.ImageFolder(data_cfg["path"], transform=tf)
    root = data_cfg.get("root") or default_data_root()
    cls = _TORCHVISION[name][0]
    return cls(str(root), train=train, download=True, transform=tf)


def get_dataloader(data_cfg: dict, train: bool = True) -> DataLoader:
    ds = get_dataset(data_cfg, train=train)
    if train and len(ds) < data_cfg.get("batch_size", 128):
        raise ValueError(f"Dataset has {len(ds)} images, fewer than batch_size={data_cfg.get('batch_size', 128)}")
    # Never more workers than CPUs (Colab has 2; asking for more only triggers a warning).
    workers = min(data_cfg.get("num_workers", 4), os.cpu_count() or 1)
    return DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 128),
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def infinite(loader):
    """Yield batches forever. Training here is counted in steps, not epochs."""
    while True:
        yield from loader
