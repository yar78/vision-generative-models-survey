#!/usr/bin/env bash
# One-time setup on a fresh RunPod GPU pod (any PyTorch template).
#
#   cd /workspace && git clone https://github.com/yar78/vision-generative-models-survey.git
#   cd vision-generative-models-survey && bash scripts/runpod/setup.sh
#
# Everything that must survive pod termination lives on the network volume
# (/workspace): the repo, datasets, run outputs, and the Hugging Face login
# (HF_HOME). Only pip packages are reinstalled on each new pod -- re-run this
# script after starting a new pod.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if [[ -d /workspace ]]; then
  export HF_HOME=/workspace/.cache/huggingface
else
  echo "WARNING: /workspace not found -- not on a RunPod pod with a network volume? Outputs will go to ./outputs." >&2
fi

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU visible" >&2

echo "== Installing genmodels (+ JupyterLab, + diffusers/peft for fine-tuning)"
pip install -q -e ".[notebook,finetune]"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "== Hugging Face login"
if python -c "from huggingface_hub import whoami; print('logged in as', whoami()['name'])" 2>/dev/null; then
  :
elif [[ -t 0 ]]; then
  echo "Not logged in. Paste a WRITE token from https://huggingface.co/settings/tokens"
  echo "(stored in ${HF_HOME:-~/.cache/huggingface}, i.e. on the network volume, so you only do this once)."
  hf auth login 2>/dev/null || huggingface-cli login
else
  echo "Not logged in. Run: HF_HOME=${HF_HOME:-~/.cache/huggingface} hf auth login   (or set HF_TOKEN)"
fi

cat <<EOF

Ready. Next:
  bash scripts/runpod/train.sh configs/ray_dog/ddpm.yaml --push-to-hub
or open the notebooks:
  HF_HOME=${HF_HOME:-} jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root notebooks/
EOF
