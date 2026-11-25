#!/usr/bin/env bash
# Terminate this pod when nothing is happening on it.
#
# Adapted from yar78/NTIRE-2026-27 scripts/pod/idle_killer.sh (written after a 4090
# sat at 0% for 49 hours). Here a DreamBooth pod finished in ~1 h and then sat idle
# until its 6-hour MAX_HOURS guard, costing ~4x the work. Started by autorun.sh.
#
#   bash scripts/runpod/idle_killer.sh              # arm it
#   bash scripts/runpod/idle_killer.sh --dry-run    # log what it WOULD do, kill nothing
#   bash scripts/runpod/idle_killer.sh --check      # evaluate once, print verdict, exit
#
# Busy = GPU >= GPU_THRESHOLD, or a download in flight, or a guarded process running
# (training, evaluation, installs), or a Claude Remote Control session active in the
# last CLAUDE_ACTIVE_MINUTES (talking to the agent uses no GPU), or the override file.
#
# Env:
#   IDLE_MINUTES     idle streak before terminating      (default 30)
#   GPU_THRESHOLD    %% utilisation counted as idle       (default 5)
#   SAMPLE_SECONDS   seconds between samples             (default 60)
#   RX_THRESHOLD_MB  network RX per sample counted busy  (default 50)
#   IDLE_SAMPLES     override the sample count, for testing
#   CLAUDE_PROJECTS_DIR   where Claude Code writes session transcripts (default ~/.claude/projects)
#   CLAUDE_ACTIVE_MINUTES a transcript written this recently counts as busy (default 2)
#   RUNPOD_POD_ID    required to actually terminate (set by Runpod on every pod)
#   RUNPOD_API_KEY   used for the REST call if set; otherwise runpodctl (which worked for
#                    this repo's MAX_HOURS guard on 2026-11-23 and 11-24) is used
#
# Escape hatch — no restart needed, checked every sample:
#   touch /workspace/.no-idle-kill
set -uo pipefail

IDLE_MINUTES="${IDLE_MINUTES:-30}"
GPU_THRESHOLD="${GPU_THRESHOLD:-5}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-60}"
RX_THRESHOLD_MB="${RX_THRESHOLD_MB:-50}"
VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
OVERRIDE_FILE="${VOLUME_ROOT}/.no-idle-kill"
CLAUDE_PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
CLAUDE_ACTIVE_MINUTES="${CLAUDE_ACTIVE_MINUTES:-2}"

# Processes that mean "busy" even at 0% GPU: this repo's jobs (training, the
# memorisation check, building the face dataset), installs and downloads.
# Deliberately NOT matching bare "python" (Jupyter would keep the pod alive forever)
# or "claude" (Remote Control runs all the time; its *activity* is checked separately).
BUSY_PATTERN="${BUSY_PATTERN:-torchrun|accelerate launch|train\\.py|memorization\\.py|make_face_dataset\\.py|pip install|apt-get|git clone|rsync|wget|unzip|hf download|huggingface-cli download}"

DRY_RUN=0
CHECK_ONCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --check)   CHECK_ONCE=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# --------------------------------------------------------------- the signals
# Max GPU utilisation across all cards. Missing nvidia-smi reports 0 rather
# than erroring, so a CPU-only pod is still killable.
gpu_util_max() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | sort -rn | head -n1 | grep -E '^[0-9]+$' || echo 0
}

# Cumulative RX bytes across real interfaces, so we can diff between samples.
rx_bytes_total() {
    local total=0 n
    for f in /sys/class/net/*/statistics/rx_bytes; do
        n="$(basename "$(dirname "$(dirname "$f")")")"
        [ "$n" = "lo" ] && continue
        total=$(( total + $(cat "$f" 2>/dev/null || echo 0) ))
    done
    echo "$total"
}

busy_process() {
    # Exclude our own process and every ancestor. Without this the killer reads
    # its own command line -- and the shell that launched it -- sees the pattern
    # text, and concludes the pod is busy forever. That failure is silent and
    # costs exactly what it was written to prevent.
    local exclude pid procs rest
    exclude=":$$:"
    pid=$(ps -o ppid= -p $$ 2>/dev/null | tr -d ' ')
    while [ -n "${pid:-}" ] && [ "$pid" -gt 1 ] 2>/dev/null; do
        exclude="${exclude}${pid}:"
        pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    done

    procs=""
    while read -r pid rest; do
        [ -z "${pid:-}" ] && continue
        case "$exclude" in *":${pid}:"*) continue ;; esac
        procs="${procs}${rest}
"
    done < <(ps -eo pid=,args= 2>/dev/null)

    printf '%s' "$procs" | grep -Eiq "$BUSY_PATTERN"
}

# A Claude Code session writes its transcript as the conversation goes; a recent write
# means someone is talking to the agent on this pod (e.g. through Remote Control).
claude_active() {
    [ -d "$CLAUDE_PROJECTS_DIR" ] || return 1
    [ -n "$(find -L "$CLAUDE_PROJECTS_DIR" -name '*.jsonl' -mmin "-${CLAUDE_ACTIVE_MINUTES}" -print -quit 2>/dev/null)" ]
}

# ------------------------------------------------------------- the terminate
terminate_pod() {
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY RUN: would terminate pod ${RUNPOD_POD_ID:-<unset>} now."
        return 0
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        log "CANNOT TERMINATE: RUNPOD_POD_ID unset (not on a Runpod pod?)."
        log "The pod is idle and will keep billing. Kill it from the console."
        return 1
    fi

    log "terminating pod ${RUNPOD_POD_ID}"

    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        local code
        code=$(curl -s -o /tmp/idlekill.out -w '%{http_code}' -X DELETE \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" 2>/dev/null)
        if [ "$code" = "200" ] || [ "$code" = "204" ]; then
            log "terminated via REST API (HTTP ${code})"
            return 0
        fi
        log "REST API returned HTTP ${code}: $(head -c 300 /tmp/idlekill.out 2>/dev/null)"
    fi

    # runpodctl ships in RunPod's own images, so it is a real fallback.
    if command -v runpodctl >/dev/null 2>&1; then
        log "falling back to runpodctl"
        runpodctl remove pod "${RUNPOD_POD_ID}" && { log "terminated via runpodctl"; return 0; }
    fi

    log "ALL TERMINATION METHODS FAILED. Kill this pod from the console."
    return 1
}

# ---------------------------------------------------------------- the verdict
# Returns 0 for idle, 1 for busy, and logs the reason either way.
evaluate() {
    local gpu rx_now rx_delta_mb
    gpu="$(gpu_util_max)"
    rx_now="$(rx_bytes_total)"
    rx_delta_mb=$(( (rx_now - ${PREV_RX:-$rx_now}) / 1024 / 1024 ))
    PREV_RX="$rx_now"

    if [ -f "$OVERRIDE_FILE" ]; then
        log "BUSY: override file present (${OVERRIDE_FILE})"; return 1
    fi
    if [ "$gpu" -ge "$GPU_THRESHOLD" ]; then
        log "BUSY: gpu ${gpu}% >= ${GPU_THRESHOLD}%"; return 1
    fi
    if [ "$rx_delta_mb" -ge "$RX_THRESHOLD_MB" ]; then
        log "BUSY: network +${rx_delta_mb} MB this sample (download in flight)"; return 1
    fi
    if busy_process; then
        log "BUSY: a guarded process is running"; return 1
    fi
    if claude_active; then
        log "BUSY: Claude session active in the last ${CLAUDE_ACTIVE_MINUTES} min"; return 1
    fi
    log "IDLE: gpu ${gpu}%, network +${rx_delta_mb} MB, no guarded process, no Claude activity"
    return 0
}

# --------------------------------------------------------------------- main
# IDLE_SAMPLES exists so you can prove the terminate path works without
# waiting 30 minutes -- a week-1 exit criterion is "idle-killer tested".
required_idle_samples="${IDLE_SAMPLES:-$(( (IDLE_MINUTES * 60) / SAMPLE_SECONDS ))}"
[ "$required_idle_samples" -lt 1 ] && required_idle_samples=1

if [ "$CHECK_ONCE" = "1" ]; then
    evaluate && log "verdict: IDLE" || log "verdict: BUSY"
    exit 0
fi

idle_window_desc="$(( required_idle_samples * SAMPLE_SECONDS ))s"
log "idle-killer armed: terminate after ${idle_window_desc} idle" \
    "(${required_idle_samples} samples x ${SAMPLE_SECONDS}s, gpu<${GPU_THRESHOLD}%)"
[ "$DRY_RUN" = "1" ] && log "DRY RUN — nothing will actually be terminated"
log "pause anytime with: touch ${OVERRIDE_FILE}"

idle_streak=0
PREV_RX="$(rx_bytes_total)"
sleep "$SAMPLE_SECONDS"

while true; do
    if evaluate; then
        idle_streak=$(( idle_streak + 1 ))
        remaining=$(( (required_idle_samples - idle_streak) * SAMPLE_SECONDS / 60 ))
        if [ "$idle_streak" -ge "$required_idle_samples" ]; then
            log "idle for ${idle_window_desc} — terminating"
            terminate_pod
            exit 0
        fi
        # Only shout near the end; no point logging every minute of a quiet night.
        if [ "$remaining" -le 10 ]; then
            log "  -> ${remaining} min until termination (streak ${idle_streak}/${required_idle_samples})"
        fi
    else
        [ "$idle_streak" -gt 0 ] && log "  -> idle streak reset"
        idle_streak=0
    fi
    sleep "$SAMPLE_SECONDS"
done
