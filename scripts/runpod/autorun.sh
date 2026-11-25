#!/usr/bin/env bash
# Unattended pod bootstrap: set up, then train a list of configs once Hugging Face is logged in.
#
# Meant to be launched by the pod's start command (see scripts/runpod/README.md), e.g.
#   AUTORUN_CONFIGS="configs/ray_dog_faces/ddpm.yaml,configs/ray_dog_faces/flow_matching.yaml" \
#   MAX_HOURS=10 nohup bash scripts/runpod/autorun.sh > /workspace/autorun.log 2>&1 &
#
#   AUTORUN_CONFIGS  comma-separated configs, trained side by side on the one GPU, each
#                    pushed to the Hub when done (needs a Hugging Face login: training
#                    waits for one -- run scripts/runpod/claude_remote.sh in the pod terminal)
#   AUTORUN_COMMANDS optional shell commands, separated by ';;', run one after another in the
#                    repo (after the Hugging Face login, before training), output appended
#                    to /workspace/autorun.log -- e.g. an evaluation script
#   MAX_HOURS        hard cost cap: the pod terminates itself this long after boot, whatever
#                    it is doing (0 = never). Outputs are pushed to the Hub as runs finish.
#   IDLE_MINUTES     the pod terminates itself after this long idle (default 30; 0 = off):
#                    no GPU use, no job or download running, no Claude agent activity.
#                    See scripts/runpod/idle_killer.sh; pause with: touch /workspace/.no-idle-kill
#
# Progress: /workspace/autorun.log, each run's train.log, TensorBoard on port 6006,
# and /workspace/AUTORUN_STATUS (one line: waiting-for-hf-login / training / done).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
STATUS=/workspace/AUTORUN_STATUS
status() { echo "$1" > "$STATUS"; echo "$(date -Is) status: $1"; }
export HF_HOME=/workspace/.cache/huggingface
export GENMODELS_OUTPUT=/workspace/genmodels/outputs GENMODELS_DATA=/workspace/genmodels/data

# ---- cost guard ----------------------------------------------------------------------
if [[ "${MAX_HOURS:-0}" != "0" ]]; then
  (
    sleep "$(python3 -c "print(int(float('${MAX_HOURS}') * 3600))")"
    echo "$(date -Is) MAX_HOURS=${MAX_HOURS} reached: terminating pod ${RUNPOD_POD_ID:-?}"
    if command -v runpodctl >/dev/null && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
      runpodctl remove pod "$RUNPOD_POD_ID"
    else
      echo "runpodctl or RUNPOD_POD_ID missing: cannot self-terminate" >&2
    fi
  ) &
  echo "$(date -Is) cost guard armed: terminate after ${MAX_HOURS} h"
fi

# ---- idle guard --------------------------------------------------------------------------
# Armed at boot: waiting too long for the logins also counts as idle.
if [[ "${IDLE_MINUTES:-30}" != "0" ]]; then
  IDLE_MINUTES="${IDLE_MINUTES:-30}" nohup bash scripts/runpod/idle_killer.sh >> /workspace/idle_killer.log 2>&1 &
  echo "$(date -Is) idle guard armed: terminate after ${IDLE_MINUTES:-30} idle minutes (log: /workspace/idle_killer.log)"
fi

# ---- setup -----------------------------------------------------------------------------
status setting-up
bash scripts/runpod/setup.sh </dev/null || { status setup-failed; exit 1; }

if ! command -v tmux >/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null || echo "tmux install failed" >&2
fi
# Claude Code, with its login kept on the pod's /workspace disk.
mkdir -p /workspace/.claude-home
[[ -e /root/.claude && ! -L /root/.claude ]] && cp -a /root/.claude/. /workspace/.claude-home/ && rm -rf /root/.claude
ln -sfn /workspace/.claude-home /root/.claude
if ! command -v claude >/dev/null && [[ ! -x /root/.local/bin/claude ]]; then
  curl -fsSL https://claude.ai/install.sh | bash || echo "Claude Code install failed" >&2
fi
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ---- wait for a Hugging Face login (private datasets, --push-to-hub) -------------------
status waiting-for-hf-login
until python3 -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; do sleep 20; done
echo "$(date -Is) Hugging Face login found"

# ---- one-off commands (e.g. evaluations) ------------------------------------------------
if [[ -n "${AUTORUN_COMMANDS:-}" ]]; then
  status running-commands
  while IFS= read -r cmd; do
    cmd="$(echo "$cmd" | sed 's/^ *//; s/ *$//')"
    [[ -z "$cmd" ]] && continue
    echo "$(date -Is) running: $cmd"
    bash -c "$cmd" || echo "$(date -Is) command failed (exit $?): $cmd"
  done < <(echo "$AUTORUN_COMMANDS" | sed 's/;;/\n/g')
fi

# ---- train ----------------------------------------------------------------------------
status training
IFS=',' read -ra CONFIGS <<< "${AUTORUN_CONFIGS:-}"
for cfg in "${CONFIGS[@]}"; do
  [[ -z "$cfg" ]] && continue
  echo "$(date -Is) starting $cfg"
  bash scripts/runpod/train.sh "$cfg" --push-to-hub
done

# Wait for every training process this script started, then mark done.
sleep 30
while pgrep -f "scripts/train.py" >/dev/null; do sleep 60; done
status done
echo "$(date -Is) all runs finished; see each run's train.log for the Hub links"
