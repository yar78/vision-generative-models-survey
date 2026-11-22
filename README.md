# vision-generative-models-survey

A playground for implementing, training and comparing the main families of
image (and later video) generative models — from scratch where it teaches
something, on top of established libraries where it doesn't.

Every model trained or fine-tuned here can be pushed to the Hugging Face Hub
under my account, with its config, weights, sample grid and metrics, so any
result can be reloaded with one line.

> **Status:** planning. This README describes the intended structure; sections
> marked _(planned)_ do not exist in code yet.

---

## Model families

| Family | What it learns | Planned implementations | First dataset |
|---|---|---|---|
| **GAN** | Generator vs. discriminator minimax game | DCGAN, WGAN-GP, conditional GAN | MNIST → CIFAR-10 |
| **StyleGAN** | Style-modulated generator with a mapping network | StyleGAN2-ADA (fine-tune), small from-scratch StyleGAN2 | FFHQ-64 / AFHQ |
| **VAE** | Encoder/decoder trained on the ELBO | VAE, β-VAE, VQ-VAE | MNIST → CelebA-64 |
| **Diffusion** | Reverse a fixed noising process | DDPM, DDIM sampling, classifier-free guidance, latent diffusion (fine-tune SD) | CIFAR-10 → CelebA-64 |
| **Flow matching** | A velocity field transporting noise to data | Conditional flow matching, rectified flow, latent FM | CIFAR-10 → CelebA-64 |

Starting on small datasets (MNIST, CIFAR-10, 64×64 faces) keeps a full training
run in the range of minutes to a few hours on a single GPU, so the families
can be compared on the same data before anything is scaled up.

## Planned layout _(planned)_

```
genmodels/
  common/          data loaders, EMA, logging, checkpointing, seeding
  hub/             Hugging Face Hub upload/download + model-card rendering
  gan/             DCGAN, WGAN-GP, cGAN
  stylegan/        StyleGAN2 (+ ADA)
  vae/             VAE, beta-VAE, VQ-VAE
  diffusion/       DDPM/DDIM, CFG, latent diffusion
  flow_matching/   CFM, rectified flow
configs/           one YAML per experiment  (e.g. configs/diffusion/ddpm_cifar10.yaml)
scripts/
  train.py         python scripts/train.py --config <yaml> [--push-to-hub]
  sample.py        generate a grid / batch from a local or Hub checkpoint
  evaluate.py      FID / IS / precision-recall against a reference set
notebooks/         exploration and visual comparisons
tests/             shape and one-step smoke tests per model
```

Each family follows the same contract — a `nn.Module` plus a config — so
`train.py`, `sample.py` and `evaluate.py` work unchanged across all of them.

## Hugging Face integration

### One-time setup

1. Create a **write** token at <https://huggingface.co/settings/tokens>.
2. Make it available without committing it, either:
   ```bash
   huggingface-cli login            # stores the token in ~/.cache/huggingface
   ```
   or put `HF_TOKEN=hf_...` in a local `.env` (git-ignored).
3. Set the account/org the models go to:
   ```bash
   export HF_USERNAME=<your-hf-username>
   ```

The token never goes into git, configs, or model cards.

### What gets uploaded _(planned)_

Each training run can end with `--push-to-hub`, which creates (or updates) a
model repo named

```
<HF_USERNAME>/<family>-<model>-<dataset>[-<tag>]
e.g.  <HF_USERNAME>/diffusion-ddpm-cifar10
      <HF_USERNAME>/flowmatching-rectified-celeba64-v2
```

containing:

| File | Purpose |
|---|---|
| `model.safetensors` | weights (EMA weights where the model uses EMA) |
| `config.yaml` | the exact experiment config used to train it |
| `README.md` | generated model card: family, dataset, steps, hardware, metrics, git commit |
| `samples.png` | fixed-seed sample grid at the final checkpoint |
| `metrics.json` | FID and other evaluation numbers, with the reference set they were computed against |

Models are built on `huggingface_hub.PyTorchModelHubMixin`, so anything pushed
can be reloaded directly:

```python
from genmodels.diffusion import DDPM
model = DDPM.from_pretrained("<HF_USERNAME>/diffusion-ddpm-cifar10")
```

Fine-tunes of existing pipelines (e.g. Stable Diffusion LoRAs) use the
`diffusers` `push_to_hub` path instead, so they load with the standard
`diffusers` API. Repos are created **private** by default; making one public is
a deliberate step.

## Getting started _(planned)_

```bash
git clone https://github.com/yar78/vision-generative-models-survey.git
cd vision-generative-models-survey
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies: PyTorch, torchvision, `huggingface_hub`, `diffusers`,
`accelerate`, `safetensors`, `torch-fidelity` (FID), and an experiment logger
(TensorBoard or Weights & Biases).

```bash
python scripts/train.py --config configs/gan/dcgan_mnist.yaml
python scripts/train.py --config configs/diffusion/ddpm_cifar10.yaml --push-to-hub
python scripts/sample.py --from-hub <HF_USERNAME>/diffusion-ddpm-cifar10 --n 64
```

## Evaluation

So that the families can be compared fairly:

- **Same data, same resolution** for every model in a comparison.
- **FID** (50k samples where affordable, otherwise stated) against the
  training-set statistics, computed with one fixed tool (`torch-fidelity`).
- **Fixed-seed sample grids** saved for every checkpoint that gets reported.
- **Cost is part of the result:** training GPU-hours and sampling time per
  image are recorded next to quality, since that is where the families differ
  most (one-step GANs vs. many-step diffusion vs. few-step flow matching).

## Roadmap

1. Repo scaffolding: package layout, config system, shared training loop, Hub module.
2. VAE on MNIST — simplest end-to-end path, used to validate the Hub upload/reload loop.
3. DCGAN → WGAN-GP on CIFAR-10.
4. DDPM (+ DDIM sampling) on CIFAR-10.
5. Conditional flow matching / rectified flow on CIFAR-10; compare against DDPM at equal compute.
6. StyleGAN2-ADA on a 64×64 faces dataset.
7. Latent variants: VQ-VAE as a tokenizer, latent diffusion, latent flow matching.
8. Fine-tuning pretrained models (Stable Diffusion / SDXL LoRA, FLUX-style flow models).
9. Extension to video generation.

## Repository rules

- **No data or weights in git.** Datasets, checkpoints and sample dumps are
  git-ignored; weights live on the Hugging Face Hub.
- **No secrets in git.** `HF_TOKEN`, `WANDB_API_KEY` etc. stay in the
  environment or a local `.env`.
- **Every reported number is reproducible:** its config, git commit and Hub repo
  are recorded together.

## References

- Goodfellow et al., *Generative Adversarial Networks*, 2014
- Radford et al., *DCGAN*, 2015 · Gulrajani et al., *WGAN-GP*, 2017
- Karras et al., *StyleGAN2*, 2020 · *StyleGAN2-ADA*, 2020
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013 · van den Oord et al., *VQ-VAE*, 2017
- Ho et al., *DDPM*, 2020 · Song et al., *DDIM*, 2020 · Rombach et al., *Latent Diffusion*, 2022
- Lipman et al., *Flow Matching for Generative Modeling*, 2022 · Liu et al., *Rectified Flow*, 2022
