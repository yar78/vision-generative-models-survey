---
name: run-pod
description: Launch, monitor and shut down Runpod GPU pods for this repo (training, evaluations, DreamBooth), always with the idle guard (terminate after 30 idle minutes) and a Claude Code agent on the pod via Remote Control. Use whenever the user asks to run something on a GPU / Runpod / "a pod", to check on or stop a pod, or when a job here is too slow for CPU.
---

# run-pod

Every pod goes through **`scripts/runpod/pod.py`**. Never create a pod any other way (raw
API call, console instructions): `pod.py create` is what guarantees the cost guards.

## What a pod gets (all automatic)

`pod.py create` starts the pod with `scripts/runpod/autorun.sh` as its boot job, which:

1. arms the **idle guard**, `scripts/runpod/idle_killer.sh`: terminate after `--idle-minutes`
   (default 30; `pod.py` refuses less than 10 and cannot turn it off) with GPU < 5%, no
   training / evaluation / install / download process, and no conversation with the pod's
   Claude agent in the last 2 minutes. Log: `/workspace/idle_killer.log`. Pause on the pod
   with `touch /workspace/.no-idle-kill`.
2. arms the **hard cap**: terminate `--max-hours` after boot (default 8).
3. installs the repo (`scripts/runpod/setup.sh`) and Claude Code (login kept on `/workspace`).
4. starts **Claude Remote Control** in tmux (`claude-rc`) if Claude is already logged in.
5. logs in to Hugging Face from the Runpod secret `huggingface_token` if the account has
   one (else waits for the manual login), runs `--commands` in order, then trains
   `--configs` side by side with `--push-to-hub`. Status line: `/workspace/AUTORUN_STATUS`.
   Hub details (tokens, uploads, updates, reading results): the **`hugging-face`** skill.

## Procedure

1. **Access.** `python scripts/runpod/pod.py balance`. If it fails with an auth error, this
   environment has no Runpod key: call `request_credential` for host **`api.runpod.io`**
   (bearer), asking the user for an API key from console.runpod.io/user/settings (All
   access). Use GraphQL only: `rest.runpod.io` returns 502 through the cloud proxy.
   Locally, the key is read from `RUNPOD_API_KEY`.
2. **Budget.** Report the balance. Estimate cost = $/h × expected hours (+30 min idle
   tail). If the hard cap × price is a large share of the balance, lower `--max-hours`
   or ask first.
3. **Commit and push first.** The pod clones the current branch from GitHub at boot;
   unpushed code is not on the pod.
4. **Create.**
   ```bash
   python scripts/runpod/pod.py create --name <short-name> \
       --configs configs/... [more configs] \
       --commands "python scripts/eval/memorization.py --repo yaryar78/<model> --push" \
       --max-hours 6
   ```
   - GPUs are tried in order: Community 4090, 3090, 3090 Ti, A5000, then Secure A5000 and
     4090 (pricier; `--community-only` to forbid). All have 24 GB, enough for DreamBooth.
   - "no instances available" for every type is normal at busy times: retry later.
   - Use `--dry-run` to show the exact request without creating anything.
5. **Tell the user the one manual step** (pod.py prints it): in the Runpod console,
   Connect -> Start Web Terminal on the pod, then
   `bash /workspace/vision-generative-models-survey/scripts/runpod/claude_remote.sh`.
   It logs in to Claude -- and to Hugging Face if the pod has no `huggingface_token` secret
   (jobs wait for that login; pod.py says which case applies) -- and starts Remote Control:
   the session `runpod-genmodels` appears in claude.ai/code and the Claude app. Logins
   only the account owner can do; do not try to automate them. **The idle guard counts
   from boot**: if nobody logs in within the idle window, the pod terminates itself.
6. **Monitor.** Schedule a check-in with `send_later` (about 45-60 min). On each:
   `python scripts/runpod/pod.py status <id>` (GPU %, uptime, balance) and the Hub
   (`HF_TOKEN=proxy-injected`, `HfApi().model_info(repo)`: a model repo appears when its
   run finishes). GPU at 0% with nothing new on the Hub long after boot means the login
   step was not done, or setup failed: ask the user for `cat /workspace/autorun.log`.
   Keep scheduling check-ins while work remains.
7. **Finish.** When results are on the Hub, fetch and show them (samples, metrics). The
   idle guard ends the pod 30 minutes after work stops; offer to terminate sooner with
   `python scripts/runpod/pod.py terminate <id> --yes` -- but ask first if the user may be
   using the Claude session on the pod (terminating ends it).

## Don'ts

- Don't touch the network volume `ntire-data`: it belongs to the NTIRE team project.
- Don't expose TCP ports (e.g. `22/tcp`) on Community pods: it limits the search to hosts
  with a public IP and every Community GPU came back unavailable that way.
- Don't print or store API keys or tokens; the proxy injects them.
- Don't disable or work around the idle guard. For a long interactive session on the pod,
  the user can `touch /workspace/.no-idle-kill` there themselves.

## Files

| file | role |
|---|---|
| `scripts/runpod/pod.py` | create / list / status / terminate / balance (GraphQL) |
| `scripts/runpod/autorun.sh` | pod boot job: guards, setup, Claude Code, jobs |
| `scripts/runpod/idle_killer.sh` | idle guard (`--check`, `--dry-run` for testing) |
| `scripts/runpod/claude_remote.sh` | logins + Remote Control (`--no-login` at boot) |
| `scripts/runpod/train.sh`, `setup.sh` | detached training with TensorBoard; installs |
