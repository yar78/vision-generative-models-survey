"""DreamBooth (Ruiz et al., 2022) with LoRA (Hu et al., 2021) on Stable Diffusion.

Teach a pretrained text-to-image model a new *subject* -- here, one specific dog --
bound to a rare identifier token, so prompts like "a photo of sks dog on the beach"
put *that* dog in new scenes and styles.

How it works:

* **Latent diffusion.** Stable Diffusion runs the diffusion process in the 64x64x4
  latent space of a frozen VAE (512 px images), conditioned on CLIP text
  embeddings through cross-attention. The loss is the usual noise (or
  v-) prediction MSE of notebook 04, only in latent space and with a text prompt.
* **DreamBooth.** Fine-tune on photos of the subject, all captioned with the same
  prompt containing a rare token ("a photo of sks dog"). The token has almost no
  prior meaning, so the model binds the subject's identity to it; the class noun
  ("dog") lets it reuse everything it already knows about dogs.
* **Prior preservation (optional).** Fine-tuning on one dog makes *every* dog look
  like it ("language drift"). DreamBooth counters this by generating class images
  with the original model ("a photo of a dog") and training on them too:
  L = L_instance + lambda * L_prior.
* **LoRA.** Instead of updating the 860M U-Net weights, learn a low-rank update
  W + (alpha / r) * B A on the attention projections. Rank 16 is ~3M trainable
  parameters and a ~12 MB file, and the base model stays untouched.

Only the U-Net gets LoRA layers; the VAE and text encoder are frozen. The exported
artefact is a diffusers LoRA (``pytorch_lora_weights.safetensors``) that loads
with ``pipe.load_lora_weights(repo_id)``.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common.data import infinite
from .common.trainer import BaseTrainer, _git_commit

DEFAULT_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]  # the U-Net's attention projections


def _require_diffusers():
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError("DreamBooth LoRA needs the fine-tuning extras: pip install -e '.[finetune]'") from e


class ImageFiles(Dataset):
    """Generated class images (for prior preservation), resized and centre-cropped."""

    def __init__(self, files, size):
        self.files = list(files)
        self.tf = transforms.Compose([transforms.Resize(size, antialias=True), transforms.CenterCrop(size),
                                      transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        return self.tf(Image.open(self.files[i]).convert("RGB")), 0


class Trainer(BaseTrainer):
    family = "dreambooth_lora"

    # ---- setup -------------------------------------------------------------------------
    def build(self):
        _require_diffusers()
        from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from peft import LoraConfig
        from transformers import CLIPTextModel, CLIPTokenizer

        m, t = self.model_cfg, self.train_cfg
        base = m.get("pretrained", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        kw = {"revision": m.get("revision")}
        precision = t.get("mixed_precision", "bf16") if self.device.type == "cuda" else "no"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("This GPU has no bf16 support (e.g. a Colab T4): using fp16 with gradient scaling.")
            precision = "fp16"
        self.weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]
        variant = m.get("variant") if self.weight_dtype != torch.float32 else None

        # Frozen weights are loaded directly in the low-precision dtype.
        wkw = dict(kw, variant=variant, torch_dtype=self.weight_dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", **wkw)
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae", **wkw)
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", **wkw)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler", **kw)

        for module in (self.text_encoder, self.vae, self.unet):
            module.requires_grad_(False)
            module.to(self.device)
        self.text_encoder.eval()
        self.vae.eval()

        # Add LoRA adapters to the U-Net attention layers; only these train, in fp32.
        rank = int(m.get("lora_rank", 16))
        self.unet.add_adapter(LoraConfig(
            r=rank, lora_alpha=int(m.get("lora_alpha", rank)), init_lora_weights="gaussian",
            target_modules=list(m.get("target_modules", DEFAULT_TARGETS)),
        ))
        for p in self.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # Gradient checkpointing recomputes activations in the backward pass: ~40% less
        # memory, ~25% slower. "auto" turns it on for GPUs under 20 GB (e.g. a 16 GB T4).
        gc = t.get("gradient_checkpointing", "auto")
        if gc == "auto":
            gc = self.device.type == "cuda" and torch.cuda.get_device_properties(self.device).total_memory < 20 * 2**30
            if gc:
                print("GPU has < 20 GB: enabling gradient checkpointing.")
        if gc:
            self.unet.enable_gradient_checkpointing()
        # fp16 gradients underflow without loss scaling; bf16 and fp32 do not need it.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.weight_dtype == torch.float16)

        self.lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        self.nets["unet"] = self.unet
        self.base_lr = float(t.get("lr", 1e-4))
        self.opts["lora"] = torch.optim.AdamW(self.lora_params, lr=self.base_lr,
                                              weight_decay=float(t.get("weight_decay", 1e-2)))
        self.warmup = int(t.get("warmup_steps", 100))
        self.grad_clip = float(t.get("grad_clip", 1.0))

        # Prompts are fixed, so encode them once.
        meth = self.method_cfg
        self.instance_prompt = meth.get("instance_prompt", "a photo of sks dog")
        self.class_prompt = meth.get("class_prompt", "a photo of a dog")
        self.prior = bool(meth.get("with_prior_preservation", False))
        self.prior_weight = float(meth.get("prior_loss_weight", 1.0))
        self.instance_emb = self.encode_prompt([self.instance_prompt])
        self.class_emb = self.encode_prompt([self.class_prompt])
        self.class_loader = None

        # Sampling pipeline sharing the (LoRA-equipped) U-Net. DPM-Solver++: good images in ~25 steps.
        self.pipe = StableDiffusionPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.validation_prompts = list(meth.get("validation_prompts", [self.instance_prompt]))
        self.sample_steps = int(meth.get("sample_steps", 25))
        self.guidance = float(meth.get("guidance_scale", 7.5))

    @torch.no_grad()
    def encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        ids = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    def num_params(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.lora_params)
        return {"lora (trainable)": trainable, "unet (frozen)": sum(p.numel() for p in self.unet.parameters()) - trainable}

    # ---- prior preservation ------------------------------------------------------------
    def prepare_class_images(self) -> None:
        """Generate ``num_class_images`` images of the class prompt with the *original*
        model (LoRA disabled). Done once; reused on resume and across runs of this config."""
        if not self.prior:
            return
        n = int(self.method_cfg.get("num_class_images", 200))
        out = Path(self.method_cfg.get("class_data_dir") or self.run_dir / "class_images")
        out.mkdir(parents=True, exist_ok=True)
        have = sorted(out.glob("*.png"))
        if len(have) < n:
            print(f"Generating {n - len(have)} class images for '{self.class_prompt}' -> {out}")
            self.unet.disable_adapters()
            try:
                bs = 4
                for start in range(len(have), n, bs):
                    k = min(bs, n - start)
                    imgs = self._generate([self.class_prompt] * k, seed=start)
                    for j, img in enumerate(imgs):
                        transforms.functional.to_pil_image(img).save(out / f"{start + j:05d}.png")
            finally:
                self.unet.enable_adapters()
            have = sorted(out.glob("*.png"))
        loader = DataLoader(ImageFiles(have[:n], self.image_size), batch_size=self.data_cfg.get("batch_size", 4),
                            shuffle=True, drop_last=True, num_workers=0)
        self.class_loader = infinite(loader)

    def fit(self, steps=None, progress=True):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_class_images()
        return super().fit(steps, progress)

    # ---- training ----------------------------------------------------------------------
    def diffusion_loss(self, pixels: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """The latent-diffusion objective: noise the VAE latents, predict the noise (or v)."""
        with torch.no_grad():
            latents = self.vae.encode(pixels.to(self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (len(latents),), device=self.device)
        noisy = self.noise_scheduler.add_noise(latents, noise, t)
        if self.noise_scheduler.config.prediction_type == "v_prediction":  # SD 2.x
            target = self.noise_scheduler.get_velocity(latents, noise, t)
        else:
            target = noise
        with torch.autocast(self.device.type, dtype=self.weight_dtype, enabled=self.weight_dtype != torch.float32):
            pred = self.unet(noisy, t, text_emb.expand(len(latents), -1, -1)).sample
        return F.mse_loss(pred.float(), target.float())

    def training_step(self, x, y):
        for g in self.opts["lora"].param_groups:
            g["lr"] = self.base_lr * min(1.0, (self.step + 1) / max(1, self.warmup))
        loss = inst = self.diffusion_loss(x, self.instance_emb)
        logs = {"loss_instance": inst.item()}
        if self.class_loader is not None:
            cx, _ = next(self.class_loader)
            prior = self.diffusion_loss(cx.to(self.device), self.class_emb)
            loss = inst + self.prior_weight * prior
            logs["loss_prior"] = prior.item()
        opt = self.opts["lora"]
        opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.lora_params, self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()
        logs["loss"] = loss.item()
        return logs

    # ---- sampling ----------------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, prompts: list[str], seed: int | None = None, steps: int | None = None,
                  guidance_scale: float | None = None, lora_scale: float = 1.0) -> torch.Tensor:
        """Run the pipeline; returns images in [0, 1], shape (n, 3, H, W), on CPU."""
        gen = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        with torch.autocast(self.device.type, dtype=self.weight_dtype, enabled=self.weight_dtype != torch.float32):
            out = self.pipe(prompts, num_inference_steps=steps or self.sample_steps,
                            guidance_scale=self.guidance if guidance_scale is None else guidance_scale,
                            height=self.image_size, width=self.image_size, generator=gen, output_type="pt",
                            cross_attention_kwargs={"scale": lora_scale})
        return out.images.float().cpu()

    @torch.no_grad()
    def sample(self, n, prompts: list[str] | None = None, **kwargs):
        """``n`` images cycling through the validation prompts, in [-1, 1] (for grids)."""
        prompts = prompts or self.validation_prompts
        prompts = [prompts[i % len(prompts)] for i in range(n)]
        imgs = []
        for i in range(0, n, 4):  # small batches keep 512-px sampling within memory
            imgs.append(self._generate(prompts[i:i + 4], **kwargs))
        return torch.cat(imgs) * 2 - 1

    @torch.no_grad()
    def sample_fixed(self, n: int, seed: int = 1234, **kwargs):
        return self.sample(n, seed=seed, **kwargs)

    # ---- checkpoints: only the LoRA weights + optimizer, not the 3.4 GB U-Net ----------
    def lora_state_dict(self):
        from peft.utils import get_peft_model_state_dict

        return get_peft_model_state_dict(self.unet)

    def save_checkpoint(self) -> None:
        state = {"step": self.step, "train_seconds": self._train_seconds, "lora": self.lora_state_dict(),
                 "opts": {k: v.state_dict() for k, v in self.opts.items()}, "history": self.history}
        tmp = self.ckpt_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def load_checkpoint(self, path=None) -> None:
        from peft.utils import set_peft_model_state_dict

        state = torch.load(path or self.ckpt_path, map_location=self.device, weights_only=False)
        set_peft_model_state_dict(self.unet, state["lora"])
        for k, v in state["opts"].items():
            self.opts[k].load_state_dict(v)
        self.step, self._train_seconds = state["step"], state.get("train_seconds", 0.0)
        self.history = state.get("history", [])

    # ---- export ------------------------------------------------------------------------
    def export_model(self):
        raise NotImplementedError("DreamBooth LoRA exports diffusers LoRA weights; see export()")

    def export(self) -> Path:
        """Write ``model/pytorch_lora_weights.safetensors`` (diffusers format) plus the
        same ``metrics.json`` / ``run_info.json`` as the other families."""
        from diffusers import StableDiffusionPipeline
        from diffusers.utils import convert_state_dict_to_diffusers

        StableDiffusionPipeline.save_lora_weights(
            self.run_dir / "model", unet_lora_layers=convert_state_dict_to_diffusers(self.lora_state_dict()),
            safe_serialization=True,
        )
        metrics = {}
        if self.history:
            tail = self.history[-max(1, len(self.history) // 10):]
            keys = [k for k in tail[-1] if k != "step"]
            metrics = {f"final_{k}": sum(h[k] for h in tail if k in h) / len(tail) for k in keys}
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        info = {
            "family": self.family,
            "model_class": "diffusers.StableDiffusionPipeline",
            "base_model": self.model_cfg.get("pretrained", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt if self.prior else None,
            "lora_rank": int(self.model_cfg.get("lora_rank", 16)),
            "name": self.cfg["name"],
            "dataset": self.data_cfg.get("hf_repo") or self.data_cfg.get("dataset"),
            "image_size": self.image_size,
            "batch_size": self.data_cfg.get("batch_size", 4),
            "steps": self.step,
            "train_hours": round(self._train_seconds / 3600, 3),
            "params": self.num_params(),
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else platform.processor() or self.device.type,
            "git_commit": _git_commit(),
        }
        (self.run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        return self.run_dir
