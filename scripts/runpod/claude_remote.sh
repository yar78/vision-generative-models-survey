#!/usr/bin/env bash
# One-time, interactive: log the pod in to Hugging Face and Claude, then start a Claude
# Remote Control session you can drive from claude.ai/code or the Claude app.
#
# Run in the pod's web terminal (Connect -> Start Web Terminal):
#   bash /workspace/vision-generative-models-survey/scripts/runpod/claude_remote.sh
#
# Both logins are stored on the pod's /workspace disk. Remote Control runs inside tmux
# (session "claude-rc"), so it keeps running after you close the terminal; reattach with
#   tmux attach -t claude-rc
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

echo "== Hugging Face"
if python3 -c "from huggingface_hub import whoami; print('logged in as', whoami()['name'])" 2>/dev/null; then
  :
else
  echo "Paste a WRITE token from https://huggingface.co/settings/tokens"
  hf auth login 2>/dev/null || huggingface-cli login
  echo "(training starts automatically now -- see /workspace/autorun.log)"
fi

echo "== Claude"
command -v claude >/dev/null || { echo "Claude Code is not installed yet -- wait for /workspace/autorun.log to pass setup, then re-run." >&2; exit 1; }
if ! claude auth status >/dev/null 2>&1; then
  echo "Sign in with the Claude account (subscription) whose app should show this pod's session."
  claude auth login
fi
claude auth status | head -3 || true

echo "== Remote Control"
command -v tmux >/dev/null || { echo "tmux missing: apt-get install -y tmux" >&2; exit 1; }
if tmux has-session -t claude-rc 2>/dev/null; then
  echo "already running (tmux attach -t claude-rc)"
else
  tmux new-session -d -s claude-rc -c "$REPO" \
    "export HF_HOME=$HF_HOME PATH=$PATH GENMODELS_OUTPUT=/workspace/genmodels/outputs GENMODELS_DATA=/workspace/genmodels/data; claude remote-control --name runpod-genmodels; echo 'Remote Control exited'; exec bash"
  sleep 8
  tmux capture-pane -p -t claude-rc | tail -15
fi
cat <<EOF

Done. Open claude.ai/code or the Claude app: the session "runpod-genmodels" runs on this
pod, in $REPO. Training progress: /workspace/autorun.log, TensorBoard on port 6006.
Reattach here with:  tmux attach -t claude-rc   (detach: Ctrl-b then d)
EOF
