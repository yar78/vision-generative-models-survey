# vision-generative-models-survey

A playground for implementing, training and comparing the main families of
image generative models -- written from scratch to be read, not just run.

Each family comes as two things that share one code path:

* a **notebook** that explains the theory, trains the model, visualises what it
  learnt and pushes it to the Hugging Face Hub, and
* a **command-line run** (`scripts/train.py`, or `scripts/runpod/train.sh` on a
  RunPod GPU) using the same config file.

Every trained model can be uploaded to the Hugging Face Hub with its weights,
config, generated model card, sample grid, metrics and TensorBoard logs, and
reloaded with `Model.from_pretrained("<user>/<repo>")`.

---

## Model families

| # | Notebook | Family | What's implemented | Configs |
|---|---|---|---|---|
| 1 | [`01_vae`](notebooks/01_vae.ipynb) | **VAE** | conv VAE, beta-VAE, KL warm-up; reconstructions, interpolations, latent traversals | `vae/mnist`, `ray_dog/vae` |
| 2 | [`02_gan`](notebooks/02_gan.ipynb) | **GAN** | DCGAN (non-saturating), WGAN-GP, conditional GAN, DiffAugment | `gan/dcgan_mnist`, `gan/wgangp_cifar10`, `ray_dog/dcgan` |
| 3 | [`03_stylegan`](notebooks/03_stylegan.ipynb) | **StyleGAN2** | mapping network, modulated/demodulated convs, noise inputs, style mixing, truncation, R1, minibatch-std, EMA, DiffAugment | `stylegan/mnist`, `stylegan/cifar10`, `ray_dog/stylegan` |
| 4 | [`04_diffusion`](notebooks/04_diffusion.ipynb) | **Diffusion** | DDPM, DDIM sampling, linear/cosine schedules, classifier-free guidance | `diffusion/ddpm_mnist`, `diffusion/ddpm_cifar10`, `ray_dog/ddpm` |
| 5 | [`05_flow_matching`](notebooks/05_flow_matching.ipynb) | **Flow matching** | conditional flow matching / rectified flow, Euler & Heun solvers, logit-normal t, CFG, a 2-D toy | `flow_matching/mnist`, `flow_matching/cifar10`, `ray_dog/flow_matching` |
| 6 | [`06_dreambooth_lora`](notebooks/06_dreambooth_lora.ipynb) | **Fine-tuning: DreamBooth LoRA** | LoRA on Stable Diffusion 1.5's U-Net attention, rare-token identifier, prior preservation, LoRA strength dial, drift & memorisation checks | `ray_dog/dreambooth_lora` |

Diffusion and flow matching share one U-Net (`genmodels/common/unet.py`), and their
MNIST / CIFAR-10 / Ray_dog configs are matched (same network, batch and steps), so
they can be compared directly.

Notebooks 1-5 train **from scratch**. Notebook 6 **fine-tunes a pretrained
text-to-image model** instead -- the practical way to get high-quality images of
one subject from a few hundred photos. It needs the fine-tuning extras
(`pip install -e ".[finetune]"`: diffusers, transformers, peft) and a GPU.

## Datasets

| `data.dataset` | |
|---|---|
| `mnist`, `fashion_mnist`, `cifar10` | torchvision, downloaded automatically |
| `hf` | any Hugging Face Hub image dataset: `hf_repo: <user>/<dataset>` |
| `folder` | an ImageFolder-style directory: `path: /some/dir` |

### Ray_dog

[`yaryar78/Ray_dog`](https://huggingface.co/datasets/yaryar78/Ray_dog): 258 photos
of one dog, mixed resolutions (mostly ~2000 px), every pose and background. The
`configs/ray_dog/` experiments train every family on it at 64x64:

* photos are decoded and downscaled **once** into memory, then randomly cropped
  and flipped each step (`random_crop`, `crop_scale`, `hflip`);
* both GANs use **DiffAugment** -- without it a discriminator memorises 258
  images within a few thousand steps;
* every notebook ends with a **memorisation check** (samples next to their
  nearest training photo), because with this little data diffusion and flow
  models will start copying.

### Ray_dog_faces (derived)

`yaryar78/Ray_dog_faces`: Ray's face cropped from every usable Ray_dog photo,
centred, rotated so the eyes are level, 512x512. Built by

```bash
python scripts/data/make_face_dataset.py build --out data/ray_dog_faces     # detect, align, crop
python scripts/data/make_face_dataset.py build --out data/ray_dog_faces --exclude 12,40   # after review
python scripts/data/make_face_dataset.py push --out data/ray_dog_faces --repo-id yaryar78/Ray_dog_faces
```

* **Grounding DINO** (zero-shot detection, prompted `dog face. dog eye. dog nose.`)
  finds the face; a face box counts only if an eye or nose lies inside it.
* **Alignment:** with two eyes the photo is rotated so they are level, and the nose
  decides which way is up, so sideways heads are corrected too. Otherwise the crop is upright.
* **Rejected automatically:** no visible face, face too small, or a duplicate.
  Ray_dog contains several re-framings of the same shot, whose face crops coincide.
* `contact_sheet.jpg` shows every crop with its index, for a manual pass with `--exclude`.
* Every crop's provenance (source row, scores, roll angle, padding) is a dataset
  column; `build_info.json` records the detector, thresholds, source revision and code commit.

Configs in `configs/ray_dog_faces/` (VAE, StyleGAN2, DDPM, flow matching) train on it at 64x64 with no
random crops, which would undo the alignment.

The strongest results on a dataset this size come from fine-tuning a pretrained
model: `configs/ray_dog/dreambooth_lora.yaml` / notebook 06 teaches Stable
Diffusion 1.5 the identifier `sks dog` with a ~12 MB LoRA, so prompts like
`a photo of sks dog on the beach at sunset` render Ray at 512x512 in new scenes
(~30-45 min for 2000 steps on a 24 GB GPU).

## Layout

```
genmodels/
  common/
    config.py        YAML configs + key.path=value overrides, output/data roots
    data.py          datasets (torchvision, HF Hub, folders), all scaled to [-1, 1]
    trainer.py       shared loop: TensorBoard, sample grids, checkpoints/resume, export
    unet.py          U-Net for diffusion and flow matching
    diffaugment.py   differentiable augmentation for data-efficient GANs
    utils.py         EMA, seeding, image grids, plots, nearest-neighbour check
  vae.py  gan.py  stylegan.py  diffusion.py  flow_matching.py
  dreambooth_lora.py LoRA fine-tuning of Stable Diffusion (diffusers + peft)
  hub.py             push a run to the Hub + generate its model card
configs/<family>/*.yaml, configs/ray_dog/*.yaml
notebooks/01_vae ... 06_dreambooth_lora
scripts/
  train.py           python scripts/train.py --config <yaml> [--push-to-hub] [overrides]
  push_to_hub.py     push an already-finished run directory
  runpod/setup.sh    one-time setup on a fresh pod
  runpod/train.sh    detached training + TensorBoard on a pod
  runpod/autorun.sh  unattended pod bootstrap (setup, Claude Code, jobs, cost guards)
  runpod/idle_killer.sh   terminate the pod after 30 idle minutes
  runpod/claude_remote.sh one-time logins + Claude Remote Control on the pod
  data/make_face_dataset.py  build + upload the aligned dog-face dataset
tests/
  test_smoke.py           every from-scratch config: train -> checkpoint -> export -> reload -> resume
  test_dreambooth_lora.py DreamBooth LoRA on a tiny SD: train, export, reload with diffusers, resume
```

Every family implements the same small interface (`build`, `training_step`,
`sample`, `export_model`) on top of `BaseTrainer`, and every exported model has
`.generate(n)` and `.from_pretrained(...)`.

## Quick start (local or notebook)

```bash
git clone https://github.com/yar78/vision-generative-models-survey.git
cd vision-generative-models-survey
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,finetune,dev]"
jupyter lab notebooks/
```

Command line:

```bash
python scripts/train.py --config configs/vae/mnist.yaml
python scripts/train.py --config configs/ray_dog/ddpm.yaml --push-to-hub
python scripts/train.py --config configs/gan/dcgan_mnist.yaml train.steps=2000 data.batch_size=64
```

Runs go to `outputs/<name>/` (`$GENMODELS_OUTPUT` overrides). Re-running a
command resumes from that run's checkpoint. TensorBoard:

```bash
tensorboard --logdir outputs
```

## Training on RunPod

Use a pod with a **network volume** mounted at `/workspace`; the repo, datasets,
outputs and the Hugging Face login all live there and survive the pod.

```bash
cd /workspace && git clone https://github.com/yar78/vision-generative-models-survey.git
cd vision-generative-models-survey
bash scripts/runpod/setup.sh          # every new pod: installs deps, checks GPU + HF login
bash scripts/runpod/train.sh configs/ray_dog/ddpm.yaml --push-to-hub
tail -f /workspace/genmodels/outputs/diffusion-ddpm-ray-dog-64/train.log
bash scripts/runpod/train.sh configs/ray_dog/dreambooth_lora.yaml --push-to-hub   # Stable Diffusion LoRA
```

`train.sh` runs detached (closing the terminal does not stop it), writes to
`/workspace/genmodels/outputs/<name>/`, and serves TensorBoard on port 6006 --
expose HTTP port 6006 on the pod and open `https://<pod-id>-6006.proxy.runpod.net`.

**Fully unattended, with a Claude agent on the pod:** set the pod's start command to
clone the repo and run `scripts/runpod/autorun.sh` (it sets up, installs Claude Code and
trains `AUTORUN_CONFIGS` side by side). Two cost guards terminate the pod by themselves:
**idle for 30 minutes** (`IDLE_MINUTES`: no GPU use, no job or download running and no
conversation with the pod's Claude agent; pause with `touch /workspace/.no-idle-kill`),
and a hard cap of `MAX_HOURS` after boot.
Then, once, in the pod's web terminal:

```bash
bash /workspace/vision-generative-models-survey/scripts/runpod/claude_remote.sh
```

which logs in to Hugging Face (training waits for this) and Claude, and starts
`claude remote-control` in tmux -- the pod then appears as a session in claude.ai/code
and the Claude app.

**Cost:** an idle pod bills. For unattended runs,

```bash
TERMINATE_WHEN_DONE=1 bash scripts/runpod/train.sh configs/ray_dog/stylegan.yaml --push-to-hub
```

terminates the pod when training ends (outputs are already on the volume and,
with `--push-to-hub`, on the Hub). Rough budgets on one modern GPU: MNIST configs
5-30 min, Ray_dog configs 1-3 h, DreamBooth LoRA 30-45 min (24 GB GPU, ~12 GB used at
batch 4; `train.gradient_checkpointing: true` for smaller cards), CIFAR-10
diffusion/flow configs ~10 h. Stable Diffusion's weights (~2 GB) are cached in
`HF_HOME` on the volume, so only the first pod downloads them.

## Hugging Face Hub

**Login once** with a *write* token from <https://huggingface.co/settings/tokens>:
`hf auth login` (or `huggingface-cli login`), `notebook_login()` in a notebook,
or the `HF_TOKEN` environment variable. The token is never written into the repo,
configs or model cards.

**Where it goes:** `<user>/<config name>`, e.g. `yaryar78/diffusion-ddpm-ray-dog-64`,
where `<user>` is `hub.username`, `$HF_USERNAME` or the logged-in account.
Override with `hub.repo_id` or `--repo-id`. Repos are **private** unless
`hub.private: false` or `--public`.

**What's uploaded:**

| File | |
|---|---|
| `model.safetensors`, `config.json` | weights (EMA weights where the family uses EMA) + constructor args |
| `README.md` | generated model card: dataset, resolution, steps, params, GPU, training time, git commit, metrics, usage |
| `samples.png` | fixed-seed sample grid at the final step |
| `metrics.json`, `run_info.json`, `train_config.yaml` | final losses, run metadata, the exact config |
| `runs/` | TensorBoard logs (the Hub's *Training metrics* tab) |

**Reload anywhere:**

```python
from genmodels.diffusion import DiffusionModel
model = DiffusionModel.from_pretrained("yaryar78/diffusion-ddpm-ray-dog-64")
images = model.generate(16, steps=50)   # (16, 3, 64, 64) in [-1, 1]
```

The DreamBooth LoRA is pushed as a standard **diffusers LoRA** repo instead
(`pytorch_lora_weights.safetensors`, with `base_model` and `instance_prompt` in the
model card), so it loads without this package:

```python
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("yaryar78/dreambooth-lora-sd15-ray-dog")
pipe("a photo of sks dog on the beach at sunset").images[0]
```

A finished run can also be pushed later: `python scripts/push_to_hub.py outputs/<name>`.

## Tests

```bash
pytest -q
```

Trains every config for two steps on fake data on CPU, then checks the export,
the `from_pretrained` round trip, the model card and checkpoint resume. The
DreamBooth test uses a tiny Stable Diffusion from the Hub (skipped offline or
without the `finetune` extras) and checks that the exported LoRA, loaded with
`pipe.load_lora_weights`, reproduces the trained model's output.

## Roadmap

1. ~~Package, configs, shared trainer, Hub upload~~
2. ~~VAE, DCGAN/WGAN-GP, StyleGAN2, DDPM/DDIM, flow matching -- notebooks + scripts~~
3. ~~Hugging Face datasets (Ray_dog) with DiffAugment and memorisation checks~~
4. FID / precision-recall evaluation (`torch-fidelity`) recorded in `metrics.json`
5. Latent models: VQ-VAE / KL autoencoder as a tokenizer, latent diffusion and latent flow matching
6. ~~DreamBooth LoRA on Stable Diffusion 1.5 with Ray_dog~~ -- next: SDXL, text-encoder LoRA, captions, a flow-matching base (SD3 / FLUX)
7. Reflow and distillation toward few-step generation
8. Video generation

## References

- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013 · Higgins et al., *beta-VAE*, 2017
- Goodfellow et al., *Generative Adversarial Networks*, 2014 · Radford et al., *DCGAN*, 2015 · Gulrajani et al., *WGAN-GP*, 2017
- Zhao et al., *Differentiable Augmentation for Data-Efficient GAN Training*, 2020
- Karras et al., *StyleGAN*, 2019 · *StyleGAN2*, 2020 · *StyleGAN2-ADA*, 2020
- Ho et al., *DDPM*, 2020 · Song et al., *DDIM*, 2020 · Nichol & Dhariwal, *Improved DDPM*, 2021 · Ho & Salimans, *Classifier-Free Guidance*, 2022
- Rombach et al., *Latent Diffusion*, 2022 · Ruiz et al., *DreamBooth*, 2022 · Hu et al., *LoRA*, 2021
- Lipman et al., *Flow Matching for Generative Modeling*, 2022 · Liu et al., *Rectified Flow*, 2022 · Esser et al., *Scaling Rectified Flow Transformers (SD3)*, 2024
