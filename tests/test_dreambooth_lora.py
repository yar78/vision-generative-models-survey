"""DreamBooth LoRA end to end on a tiny Stable Diffusion (hf-internal-testing), on CPU:
prior preservation, training, checkpoint/resume, diffusers LoRA export and reload, model card.

Needs the [finetune] extras and access to huggingface.co; skipped otherwise."""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from genmodels import build_trainer  # noqa: E402
from genmodels.common.config import load_config  # noqa: E402
from genmodels.hub import render_model_card  # noqa: E402

TINY = "hf-internal-testing/tiny-stable-diffusion-pipe"
CONFIG = Path(__file__).parents[1] / "configs/ray_dog/dreambooth_lora.yaml"


def tiny_cfg():
    cfg = load_config(CONFIG)
    cfg["data"].update(dataset="fake", fake_size=8, image_size=32, batch_size=2, num_workers=0, channels=3)
    cfg["model"].update(pretrained=TINY, variant=None, lora_rank=4, lora_alpha=4)
    cfg["method"].update(num_class_images=4, sample_steps=2, validation_prompts=["a photo of sks dog", "a photo of a dog"])
    cfg["train"].update(steps=3, log_every=1, sample_every=3, ckpt_every=3, n_samples=2, device="cpu")
    return cfg


@pytest.fixture(scope="module")
def tiny_available():
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(TINY)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot download {TINY}: {e}")


def test_dreambooth_lora_end_to_end(tmp_path, tiny_available):
    cfg = tiny_cfg()
    trainer = build_trainer(cfg, run_dir=tmp_path)
    n = trainer.num_params()
    assert 0 < n["lora (trainable)"] < n["unet (frozen)"]
    frozen_before = {k: v.clone() for k, v in trainer.unet.state_dict().items() if "lora" not in k}

    history = trainer.fit(progress=False)
    assert "loss_prior" in history[-1] and torch.isfinite(torch.tensor(history[-1]["loss"]))
    assert len(list((tmp_path / "class_images").glob("*.png"))) == 4

    # Only LoRA weights changed.
    after = trainer.unet.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in frozen_before.items())

    for f in ("model/pytorch_lora_weights.safetensors", "samples.png", "metrics.json", "run_info.json", "checkpoint.pt"):
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "checkpoint.pt").stat().st_size < 5_000_000  # LoRA + optimizer only, not the U-Net

    # The exported LoRA loads into a fresh pipeline with the standard diffusers API
    # and reproduces the trained model's output.
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    # The tiny repo's model_index.json names Flax classes, so load each component directly.
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    pipe = StableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(TINY, subfolder="vae"),
        text_encoder=CLIPTextModel.from_pretrained(TINY, subfolder="text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(TINY, subfolder="tokenizer"),
        unet=UNet2DConditionModel.from_pretrained(TINY, subfolder="unet"),
        scheduler=DPMSolverMultistepScheduler.from_pretrained(TINY, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    kw = dict(num_inference_steps=2, guidance_scale=7.5, height=32, width=32, output_type="pt")
    base = pipe("a photo of sks dog", generator=torch.Generator().manual_seed(0), **kw).images
    pipe.load_lora_weights(tmp_path / "model")
    tuned = pipe("a photo of sks dog", generator=torch.Generator().manual_seed(0), **kw).images
    ours = trainer._generate(["a photo of sks dog"], seed=0, steps=2)
    assert not torch.allclose(base, tuned)
    assert torch.allclose(tuned, ours, atol=1e-4)

    info = json.loads((tmp_path / "run_info.json").read_text())
    card = render_model_card(tmp_path, "someone/ray-lora")
    assert "base_model: " + TINY in card and "instance_prompt: a photo of sks dog" in card
    assert 'pipe.load_lora_weights("someone/ray-lora")' in card and info["lora_rank"] == 4

    # Resume picks up the LoRA state from the checkpoint.
    cfg["train"]["steps"] = 4
    again = build_trainer(cfg, run_dir=tmp_path)
    again.fit(progress=False)
    assert again.step == 4
