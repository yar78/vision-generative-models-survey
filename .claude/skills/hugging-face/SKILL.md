---
name: hugging-face
description: Everything Hugging Face Hub in this repo -- connecting (this cloud session, Colab, local, Runpod pods), uploading and updating models, LoRAs, datasets and evaluation reports, and reading results back. Use whenever a task pushes to, pulls from, or checks the Hub (account yaryar78), or when a Hub call fails with 401/403/404.
---

# hugging-face

Account: **`yaryar78`**. New uploads from this repo are **private** unless the user asks
otherwise; the user may change visibility on the Hub later (check, don't assume). The
account also holds repos from other projects (e.g. `ray-dog`, `DDPM-Ray-dog`,
`vit-base-oxford-iiit-pets`, the Hebrew-lyrics classifiers): never modify those. Pods are
covered by the `run-pod` skill; this skill is about the Hub side.

## 1. Connecting

| where | how it is logged in | check |
|---|---|---|
| **this Claude cloud session** | network secret "Hugging Face write token" on host `huggingface.co`, injected by the proxy | `HF_TOKEN=proxy-injected python -c "from huggingface_hub import whoami; print(whoami()['name'])"` |
| **Runpod pod** | automatic if the Runpod account has a secret **`huggingface_token`**: `pod.py create` passes it as `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`. Otherwise `claude_remote.sh` asks for the token once; it is stored under `HF_HOME=/workspace/.cache/huggingface` | `tail /workspace/autorun.log` says "Hugging Face login found (...)" |
| **Colab / local / RunPod JupyterLab** | `notebook_login()` (the notebooks' push cell), `hf auth login`, or `HF_TOKEN` | same `whoami()` |

- **In this session, always set `HF_TOKEN=proxy-injected`** for code using `huggingface_hub`,
  `datasets` or `transformers`: the library then sends an `Authorization` header, which the
  proxy overwrites with the real token. Without it, private repos answer 401/404. Never print,
  log or write the real token anywhere.
- If `whoami()` fails here, the secret is missing or expired: call `request_credential` for
  host `huggingface.co` (bearer, a **write** token from huggingface.co/settings/tokens).
  Fine-grained tokens need write access to the `yaryar78` namespace (the current one has it:
  uploads have worked).
- **Network hosts** this environment needs (already allowed): `huggingface.co`, and for file
  downloads the CDN **`us.aws.cdn.hf.co`** (every file download redirects there), plus the
  `*.xethub.hf.co` / `cdn-lfs*` hosts. A download failing with a proxy 403 on a new `*.hf.co`
  host means that host must be added (`update_environment`, exact host).
- **Private repos need a login before loading** (check visibility with the listing in section 2). In Colab the push cell's `notebook_login()`
  comes too late for `Ray_dog_faces`: log in in a cell before the data cell.
- **Pods:** to make the Hugging Face login automatic, the user creates the Runpod secret once
  (console -> Settings -> Secrets, name `huggingface_token`, value a write token). Only the
  user can do this -- the value never passes through this session.

## 2. What is on the Hub

```bash
HF_TOKEN=proxy-injected python -c "
from huggingface_hub import HfApi; api = HfApi()
for m in api.list_models(author='yaryar78'): print('model  ', m.id, 'private' if m.private else 'public')
for d in api.list_datasets(author='yaryar78'): print('dataset', d.id, 'private' if d.private else 'public')"
```

| repo | made by |
|---|---|
| model `yaryar78/<config name>` | `scripts/train.py --push-to-hub` (name = the config's `name:`) |
| model `.../memorization/` folder | `scripts/eval/memorization.py --push` (report, histogram, closest pairs) |
| model `yaryar78/dreambooth-lora-sd15-ray-dog` | DreamBooth LoRA (diffusers format) |
| dataset `yaryar78/Ray_dog` | the user's 258 photos -- source data, never modified here |
| dataset `yaryar78/Ray_dog_faces` | `scripts/data/make_face_dataset.py` |

## 3. Uploading and updating

**Models** -- `genmodels/hub.py`, `push_run(run_dir)`:
- At the end of training: `python scripts/train.py --config <cfg> --push-to-hub` (checks the
  login *before* training, so a bad token fails fast).
- A finished run later: `python scripts/push_to_hub.py <run_dir> [--repo-id ...] [--public]`.
- **Updating = pushing again.** Re-pushing a run (e.g. after training longer) makes a new
  commit in the same repo; Hub history keeps the old version. Files uploaded:
  `model.safetensors` + `config.json` (or `pytorch_lora_weights.safetensors`), a generated
  `README.md` model card, `samples.png`, `metrics.json`, `run_info.json`,
  `train_config.yaml`, TensorBoard logs under `runs/`.
- Model cards are generated from `run_info.json` (`render_model_card`): to change card text
  for all future pushes, edit `genmodels/hub.py`, not the card on the Hub.
- Extra files into an existing repo (reports, figures):
  `HfApi().upload_folder(folder_path=..., path_in_repo="subdir", repo_id=..., commit_message=...)`.
- **Never make a repo public, delete a repo, or overwrite another repo's files unless the user
  asks.** Public: `HfApi().update_repo_settings(repo_id, private=False)`.

**Datasets** -- `scripts/data/make_face_dataset.py`:
- `build` (detect, align, crop; detections cached in `<out>.detections.json`, so re-cropping
  takes seconds) -> review `contact_sheet.jpg` -> `build --exclude i,j,...` -> `push`.
- An update is `push` again to the same `--repo-id`: new commit, same dataset. Record why
  rows were dropped in `build_info.json` (`review_reasons`) -- it is uploaded with the data.
- Training on a Hub dataset: `data: {dataset: hf, hf_repo: <id>}`; `exclude_indices: [...]`
  leaves rows out without changing the dataset (used for DreamBooth: other dogs, selfies).

## 4. Reading results back

```python
import json
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
api.model_info(repo).siblings                         # file list (404 = not pushed yet / no access)
json.load(open(hf_hub_download(repo, "metrics.json")))
hf_hub_download(repo, "samples.png", local_dir=...)   # then Read the image to look at it
```
- A repo appearing is how a pod run reports "finished" (`run-pod` skill check-ins).
- Load a model: `<ModelClass>.from_pretrained(repo)` (class named in `run_info.json`);
  a LoRA: `StableDiffusionPipeline.from_pretrained(base).load_lora_weights(repo)`.
- Put downloads under the scratchpad, not in the repo (`data/` and `outputs/` are git-ignored,
  but keep the Hub as the source of truth).
