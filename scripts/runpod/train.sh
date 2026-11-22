#!/usr/bin/env bash
# Launch a training run on a RunPod GPU pod, detached from your terminal.
#
#   bash scripts/runpod/train.sh <config.yaml> [--push-to-hub] [key=value overrides...]
#
#   bash scripts/runpod/train.sh configs/ray_dog/ddpm.yaml --push-to-hub
#   bash scripts/runpod/train.sh configs/gan/dcgan_mnist.yaml train.steps=5000
#   TERMINATE_WHEN_DONE=1 bash scripts/runpod/train.sh configs/ray_dog/stylegan.yaml --push-to-hub
#
# What it does:
#   * runs scripts/train.py under nohup, so closing SSH / the web terminal does not stop it
#   * writes outputs to /workspace/genmodels/outputs/<name>/ (network volume -- survives the pod)
#   * logs to <run dir>/train.log      ->  tail -f that file to watch
#   * serves TensorBoard on port 6006  ->  https://<pod-id>-6006.proxy.runpod.net
#     (expose HTTP port 6006 in the pod's settings)
#   * re-running the same command resumes from the last checkpoint
#
# TERMINATE_WHEN_DONE=1 terminates the pod when training ends (success or failure),
# so an unattended run does not keep billing. Only allowed when outputs are on
# /workspace. If combined with --push-to-hub, check that you are logged in first
# (train.py refuses to start otherwise).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,24p' "$0"; exit 1
fi
CONFIG="$1"; shift

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

if [[ -d /workspace ]]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export GENMODELS_OUTPUT="${GENMODELS_OUTPUT:-/workspace/genmodels/outputs}"
  export GENMODELS_DATA="${GENMODELS_DATA:-/workspace/genmodels/data}"
else
  export GENMODELS_OUTPUT="${GENMODELS_OUTPUT:-$REPO/outputs}"
fi

NAME="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")"
RUN_DIR="$GENMODELS_OUTPUT/$NAME"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"

if [[ "${TERMINATE_WHEN_DONE:-0}" == "1" ]]; then
  [[ "$RUN_DIR" == /workspace/* ]] || { echo "TERMINATE_WHEN_DONE needs outputs on /workspace (got $RUN_DIR)" >&2; exit 1; }
  [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null || { echo "TERMINATE_WHEN_DONE needs runpodctl and RUNPOD_POD_ID (run on a RunPod pod)" >&2; exit 1; }
fi

# TensorBoard over all runs, once per pod.
if ! pgrep -f "tensorboard.*--port 6006" >/dev/null 2>&1; then
  nohup tensorboard --logdir "$GENMODELS_OUTPUT" --host 0.0.0.0 --port 6006 >"$GENMODELS_OUTPUT/tensorboard.log" 2>&1 &
fi

CMD=(python scripts/train.py --config "$CONFIG" --run-dir "$RUN_DIR" "$@")
echo "$(date -Is) ${CMD[*]}" >>"$LOG"

nohup bash -c '
  "$@" >>"'"$LOG"'" 2>&1
  status=$?
  echo "$(date -Is) training exited with status $status" >>"'"$LOG"'"
  if [[ "'"${TERMINATE_WHEN_DONE:-0}"'" == "1" ]]; then
    echo "$(date -Is) terminating pod '"${RUNPOD_POD_ID:-}"'" >>"'"$LOG"'"
    runpodctl remove pod "'"${RUNPOD_POD_ID:-}"'"
  fi
' _ "${CMD[@]}" >/dev/null 2>&1 &

echo "Started (pid $!): ${CMD[*]}"
echo "  run dir     : $RUN_DIR"
echo "  log         : tail -f $LOG"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "  TensorBoard : https://${RUNPOD_POD_ID}-6006.proxy.runpod.net   (expose HTTP port 6006 on the pod)"
else
  echo "  TensorBoard : http://localhost:6006"
fi
[[ "${TERMINATE_WHEN_DONE:-0}" == "1" ]] && echo "  The pod will be TERMINATED when training ends."
exit 0
