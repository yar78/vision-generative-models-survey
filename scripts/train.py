#!/usr/bin/env python
"""Train any family from a config, optionally push the result to the Hugging Face Hub.

    python scripts/train.py --config configs/vae/mnist.yaml
    python scripts/train.py --config configs/diffusion/ddpm_mnist.yaml --push-to-hub
    python scripts/train.py --config configs/gan/dcgan_mnist.yaml train.steps=500 data.batch_size=64

Trailing ``key.path=value`` arguments override the config. Re-running the same
command resumes from the run directory's checkpoint.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genmodels import build_trainer  # noqa: E402
from genmodels.common.config import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", help="default: $GENMODELS_OUTPUT/<name>, /workspace/genmodels/outputs/<name> or ./outputs/<name>")
    p.add_argument("--device", help="cuda / cpu / mps (default: auto)")
    p.add_argument("--push-to-hub", action="store_true", help="upload the finished run to the Hub")
    p.add_argument("--repo-id", help="override the Hub repo id (default: <HF_USERNAME>/<name>)")
    p.add_argument("--public", action="store_true", help="create the Hub repo as public (default private)")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.push_to_hub:  # fail now, not after hours of training
        from huggingface_hub import whoami

        try:
            user = whoami()["name"]
        except Exception as e:  # noqa: BLE001
            sys.exit(f"--push-to-hub given but not logged in to Hugging Face ({e}).\n"
                     "Run `huggingface-cli login` or set HF_TOKEN first.")
        print(f"Logged in to Hugging Face as {user}")

    trainer = build_trainer(cfg, run_dir=args.run_dir, device=args.device)
    print(f"family={cfg['family']} name={cfg['name']} device={trainer.device} run_dir={trainer.run_dir}")
    print("params:", {k: f"{v / 1e6:.2f}M" for k, v in trainer.num_params().items()})
    trainer.fit()
    print(f"Done. Samples: {trainer.run_dir / 'samples.png'}")

    if args.push_to_hub:
        from genmodels.hub import push_run

        push_run(trainer.run_dir, repo_id=args.repo_id, private=False if args.public else None)


if __name__ == "__main__":
    main()
