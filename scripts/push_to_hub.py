#!/usr/bin/env python
"""Push an already-finished run directory to the Hugging Face Hub.

    python scripts/push_to_hub.py outputs/vae-mnist
    python scripts/push_to_hub.py /workspace/genmodels/outputs/diffusion-ddpm-mnist --repo-id me/my-ddpm --public
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genmodels.hub import push_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--repo-id")
    p.add_argument("--public", action="store_true")
    p.add_argument("--no-logs", action="store_true", help="do not upload TensorBoard logs")
    args = p.parse_args()
    push_run(args.run_dir, repo_id=args.repo_id, private=False if args.public else None, include_logs=not args.no_logs)


if __name__ == "__main__":
    main()
