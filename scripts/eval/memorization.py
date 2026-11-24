#!/usr/bin/env python
"""Memorisation check: how close are a model's samples to its training images?

    python scripts/eval/memorization.py --repo yaryar78/diffusion-ddpm-ray-dog-faces-64 \\
        --repo yaryar78/flowmatching-rf-ray-dog-faces-64 --n 256 --push

For each model (loaded from the Hub, or a local run's ``model/`` folder):

1. Generate ``--n`` samples (fixed seed).
2. For every sample, find its nearest training image in two spaces:
   * **DINOv2 features** (cosine similarity) -- "is this the same picture?", robust to
     small shifts and colour changes. The standard tool for this kind of check.
   * **Pixels** (L2 at the model's resolution), plus the ratio of nearest to
     second-nearest distance: a ratio well below 1 means one training image is a much
     closer match than any other (Carlini et al., 2023, use < 1/3 to flag copies).
3. **Baseline:** each training image vs its nearest *other* training image. Two photos
   of the same dog taken seconds apart are this similar; a sample more similar to a
   training image than that is a copy, not a new picture.

Writes ``<out>/<model>/``: report.md, summary.json, a histogram of similarities and
grids of the closest and of random sample/neighbour pairs. With ``--push`` these go
into the model repo under ``memorization/``. For diffusion models it also compares
the 50-step DDIM sampler with full 1000-step DDPM on the same seeds (``--samplers``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genmodels.common.data import get_dataset  # noqa: E402
from genmodels.common.utils import get_device, save_grid  # noqa: E402

DINO = "facebook/dinov2-small"


def load_model(ref: str, device):
    """``ref`` is a Hub repo id or a local run directory. Returns (model, run_info, train_config)."""
    import yaml

    p = Path(ref)
    if p.is_dir():
        run_info = json.loads((p / "run_info.json").read_text())
        cfg = yaml.safe_load((p / "config.yaml").read_text())
        weights = p / "model"
    else:
        from huggingface_hub import hf_hub_download

        run_info = json.loads(Path(hf_hub_download(ref, "run_info.json")).read_text())
        cfg = yaml.safe_load(Path(hf_hub_download(ref, "train_config.yaml")).read_text())
        weights = ref
    module, cls = run_info["model_class"].rsplit(".", 1)
    model = getattr(importlib.import_module(module), cls).from_pretrained(weights).to(device).eval()
    return model, run_info, cfg


@torch.no_grad()
def generate(model, family, n, device, seed=0, batch=64, **kw):
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        if family in ("diffusion", "flow_matching"):
            out.append(model.generate(k, device=device, **kw).cpu())
        else:
            out.append(model.generate(k, device=device).cpu())
    return torch.cat(out).clamp(-1, 1)


class Dino:
    def __init__(self, device):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(DINO).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x, batch=64):
        feats = []
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device)
            if b.shape[1] == 1:
                b = b.repeat(1, 3, 1, 1)
            b = F.interpolate((b + 1) / 2, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
            out = self.model(pixel_values=(b - self.mean) / self.std)
            feats.append(F.normalize(out.pooler_output, dim=-1).cpu())
        return torch.cat(feats)


def nn_stats(sample_feats, train_feats, sample_px, train_px, exclude_self=False):
    sim = sample_feats @ train_feats.T                        # cosine similarity
    d = torch.cdist(sample_px.flatten(1), train_px.flatten(1))  # pixel L2
    if exclude_self:
        sim.fill_diagonal_(-1)
        d.fill_diagonal_(float("inf"))
    top_sim, idx = sim.max(dim=1)
    two = d.topk(2, largest=False)
    ratio = two.values[:, 0] / two.values[:, 1].clamp(min=1e-8)
    return top_sim, idx, two.values[:, 0], ratio


def pct(t, q):
    return float(torch.quantile(t.float(), q))


def histogram(baseline, samples, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    bins = torch.linspace(min(baseline.min(), samples.min()).item(), 1.0, 40).numpy()
    plt.hist(baseline.numpy(), bins=bins, alpha=0.6, density=True, label="training image vs nearest other training image")
    plt.hist(samples.numpy(), bins=bins, alpha=0.6, density=True, label="sample vs nearest training image")
    plt.xlabel("DINOv2 cosine similarity to nearest training image")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def pairs_grid(samples, train, idx, order, path, k=24):
    """Rows of (sample, nearest training image) pairs, 6 pairs per row."""
    sel = order[:k]
    tiles = torch.stack([t for i in sel for t in (samples[i], train[idx[i]])])
    save_grid(tiles, path, nrow=12)


def check(ref, args, dino, device):
    model, run_info, cfg = load_model(ref, device)
    family = run_info["family"]
    name = run_info["name"]
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {ref} ({family}, {run_info['steps']} steps)")

    data_cfg = dict(cfg["data"], random_crop=False, hflip=False, num_workers=0)
    ds = get_dataset(data_cfg, train=True)
    train = torch.stack([ds[i][0] for i in range(min(len(ds), args.max_train))])
    print(f"   {len(train)} training images, generating {args.n} samples")

    kw = {}
    if family in ("diffusion", "flow_matching"):
        kw["steps"] = args.steps
    if family == "flow_matching":
        kw["solver"] = cfg.get("method", {}).get("solver", "euler")  # sample as configured
    samples = generate(model, family, args.n, device, seed=args.seed, **kw)

    tf, sf = dino(train), dino(samples)
    s_sim, s_idx, s_d, s_ratio = nn_stats(sf, tf, samples, train)
    b_sim, _, b_d, _ = nn_stats(tf, tf, train, train, exclude_self=True)

    base_p95, base_max = pct(b_sim, 0.95), float(b_sim.max())
    summary = {
        "model": ref, "family": family, "train_steps": run_info["steps"], "n_samples": args.n,
        "n_train": len(train), "sampler_steps": args.steps if kw else None, "feature_model": DINO,
        "sample_nn_similarity": {"median": pct(s_sim, 0.5), "p90": pct(s_sim, 0.9), "max": float(s_sim.max())},
        "train_nn_similarity_baseline": {"median": pct(b_sim, 0.5), "p95": base_p95, "max": base_max},
        "frac_samples_above_baseline_p95": float((s_sim > base_p95).float().mean()),
        "frac_samples_above_baseline_max": float((s_sim > base_max).float().mean()),
        "frac_samples_sim_above_0.95": float((s_sim > 0.95).float().mean()),
        "frac_pixel_ratio_below_1/3": float((s_ratio < 1 / 3).float().mean()),
        "distinct_training_images_matched": int(s_idx.unique().numel()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    histogram(b_sim, s_sim, out / "similarity_hist.png", f"{name}: are samples copies?")
    order = torch.argsort(s_sim, descending=True).tolist()
    pairs_grid(samples, train, s_idx, order, out / "closest_pairs.png")
    rand = torch.randperm(len(samples), generator=torch.Generator().manual_seed(1)).tolist()
    pairs_grid(samples, train, s_idx, rand, out / "random_pairs.png")
    save_grid(samples[:64], out / "samples.png", nrow=8)

    sampler_note = ""
    if family == "diffusion" and args.samplers:
        seeds_n = 16
        rows = []
        for sampler, steps in (("ddim", 50), ("ddim", 250), ("ddpm", None)):
            torch.manual_seed(123)
            g = model.generate(seeds_n, sampler=sampler, steps=steps or 50, device=device).cpu()
            rows.append(g)
            summary.setdefault("sampler_image_stats", {})[f"{sampler}{'-' + str(steps) if steps else '-1000'}"] = {
                "mean": float(g.mean()), "std": float(g.std()),
                "frac_saturated": float(((g.abs() > 0.98).float().mean())),
            }
        save_grid(torch.cat(rows), out / "samplers_ddim50_ddim250_ddpm1000.png", nrow=seeds_n)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        sampler_note = ("\n## Sampler comparison\n\nSame 16 seeds; rows: DDIM 50 steps, DDIM 250 steps, DDPM 1000 steps.\n\n"
                        "![samplers](samplers_ddim50_ddim250_ddpm1000.png)\n\n| sampler | mean | std | saturated pixels |\n|---|---|---|---|\n"
                        + "\n".join(f"| {k} | {v['mean']:.3f} | {v['std']:.3f} | {v['frac_saturated']:.1%} |"
                                    for k, v in summary["sampler_image_stats"].items()) + "\n")

    s = summary
    verdict = ("**Strong memorisation**: most samples are closer to a training image than any two training images are to each other."
               if s["frac_samples_above_baseline_max"] > 0.5 else
               "**Partial memorisation**: a sizeable share of samples are near-copies of training images."
               if s["frac_samples_above_baseline_p95"] > 0.25 else
               "**Little memorisation**: samples are about as far from the training set as training images are from each other.")
    report = f"""# Memorisation check: {name}

Model `{ref}` ({family}, {s['train_steps']} training steps), {s['n_samples']} samples vs {s['n_train']} training images,
similarity measured with [{DINO}](https://huggingface.co/{DINO}) features (cosine) and in pixels.

{verdict}

| | median | p90 / p95 | max |
|---|---|---|---|
| sample -> nearest training image | {s['sample_nn_similarity']['median']:.3f} | {s['sample_nn_similarity']['p90']:.3f} (p90) | {s['sample_nn_similarity']['max']:.3f} |
| baseline: training image -> nearest other training image | {s['train_nn_similarity_baseline']['median']:.3f} | {s['train_nn_similarity_baseline']['p95']:.3f} (p95) | {s['train_nn_similarity_baseline']['max']:.3f} |

* Samples more similar to their nearest training image than 95% of training-image pairs: **{s['frac_samples_above_baseline_p95']:.0%}**
* ... than *any* pair of training images: **{s['frac_samples_above_baseline_max']:.0%}**
* Samples with similarity > 0.95: {s['frac_samples_sim_above_0.95']:.0%}; pixel nearest/second-nearest ratio < 1/3: {s['frac_pixel_ratio_below_1/3']:.0%}
* Distinct training images that are some sample's nearest neighbour: {s['distinct_training_images_matched']} of {s['n_train']}

![histogram](similarity_hist.png)

**Closest pairs** (sample, nearest training image), most similar first:

![closest](closest_pairs.png)

**Random pairs:**

![random](random_pairs.png)
{sampler_note}"""
    (out / "report.md").write_text(report)
    print(f"   {verdict.split('**')[1]}: {s['frac_samples_above_baseline_p95']:.0%} above baseline p95, "
          f"{s['frac_samples_above_baseline_max']:.0%} above baseline max -> {out}")

    if args.push and not Path(ref).is_dir():
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(out), path_in_repo="memorization", repo_id=ref,
                              commit_message="Add memorisation check")
        print(f"   pushed to https://huggingface.co/{ref}/tree/main/memorization")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", action="append", required=True, help="Hub model repo or local run dir (repeatable)")
    p.add_argument("--n", type=int, default=256, help="samples per model")
    p.add_argument("--steps", type=int, default=50, help="sampler steps for diffusion / flow matching")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=10_000)
    p.add_argument("--samplers", action="store_true", help="diffusion: also compare DDIM 50 / 250 and DDPM 1000")
    p.add_argument("--out", default="outputs/memorization")
    p.add_argument("--push", action="store_true", help="upload each report into its model repo")
    p.add_argument("--device")
    args = p.parse_args()

    device = get_device(args.device)
    dino = Dino(device)
    results = [check(ref, args, dino, device) for ref in args.repo]
    print(json.dumps([{k: r[k] for k in ("model", "frac_samples_above_baseline_p95", "frac_samples_above_baseline_max")}
                      for r in results], indent=2))


if __name__ == "__main__":
    main()
