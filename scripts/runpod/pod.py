#!/usr/bin/env python
"""Launch and manage Runpod GPU pods for this repo -- always with the cost guards on.

    python scripts/runpod/pod.py balance
    python scripts/runpod/pod.py list
    python scripts/runpod/pod.py create --name ray-ddpm --configs configs/ray_dog_faces/ddpm.yaml
    python scripts/runpod/pod.py create --name evals --commands "python scripts/eval/memorization.py --repo me/x --push"
    python scripts/runpod/pod.py status <pod-id>
    python scripts/runpod/pod.py terminate <pod-id> --yes

Every pod created here boots into scripts/runpod/autorun.sh, which:

* arms the **idle guard** (scripts/runpod/idle_killer.sh): terminate after --idle-minutes
  (default 30, minimum 10, cannot be turned off from here) with no GPU use, no job or
  download running and no conversation with the pod's Claude agent;
* arms the **hard cap**: terminate --max-hours after boot (default 8);
* installs the repo and Claude Code, waits for a Hugging Face login, runs --commands
  one after another, then trains --configs side by side (each pushed to the Hub);
* starts **Claude Remote Control** in tmux as soon as Claude is logged in on the pod.

The one manual step -- logins that only the account owner can do -- is printed after
`create`: in the pod's web terminal, run scripts/runpod/claude_remote.sh.

Auth: the Runpod API key is read from RUNPOD_API_KEY. In a Claude Code cloud session the
environment's network proxy injects it for api.runpod.io instead, so no variable is needed
there. Uses Runpod's GraphQL API (in that environment the REST API answers 502).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"
REPO_URL = "https://github.com/yar78/vision-generative-models-survey.git"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Tried in order until one deploys. Community Cloud is cheaper; only Secure Cloud can attach
# network volumes (not needed: results go to the Hub). All have 24 GB, enough for DreamBooth.
DEFAULT_GPUS = [
    ("COMMUNITY", "NVIDIA GeForce RTX 4090"),
    ("COMMUNITY", "NVIDIA GeForce RTX 3090"),
    ("COMMUNITY", "NVIDIA GeForce RTX 3090 Ti"),
    ("COMMUNITY", "NVIDIA RTX A5000"),
    ("SECURE", "NVIDIA RTX A5000"),
    ("SECURE", "NVIDIA GeForce RTX 4090"),
]
MIN_IDLE_MINUTES = 10


def gql(query: str, variables: dict | None = None) -> dict:
    key = os.environ.get("RUNPOD_API_KEY") or "proxy-injected"  # cloud-session proxy replaces it
    req = urllib.request.Request(
        API, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        sys.exit(f"Runpod API HTTP {e.code}: {body}")
    if out.get("errors") and not out.get("data"):
        sys.exit(f"Runpod API error: {out['errors'][0].get('message')}")
    return out


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


# ---- commands --------------------------------------------------------------------------
def cmd_balance(args):
    me = gql("{ myself { clientBalance currentSpendPerHr } }")["data"]["myself"]
    print(f"balance ${me['clientBalance']:.2f}, spending ${me['currentSpendPerHr']:.3f}/h")


def cmd_list(args):
    pods = gql("{ myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } runtime { uptimeInSeconds } } } }")
    pods = pods["data"]["myself"]["pods"]
    if not pods:
        print("no pods")
    for p in pods:
        up = (p.get("runtime") or {}).get("uptimeInSeconds")
        print(f"{p['id']}  {p['name']:<28} {p['desiredStatus']:<9} ${p['costPerHr']}/h  "
              f"{(p.get('machine') or {}).get('gpuDisplayName', '?'):<10} up {up // 60 if up else '-'} min")


def cmd_status(args):
    q = """query($id: String!) { myself { clientBalance } pod(input: {podId: $id}) {
        id name desiredStatus costPerHr lastStatusChange machine { gpuDisplayName location }
        runtime { uptimeInSeconds container { cpuPercent memoryPercent } gpus { gpuUtilPercent memoryUtilPercent } } } }"""
    d = gql(q, {"id": args.pod_id})["data"]
    p = d["pod"]
    if not p:
        sys.exit(f"pod {args.pod_id} not found (terminated?)")
    rt = p.get("runtime") or {}
    gpu = (rt.get("gpus") or [{}])[0]
    print(json.dumps({
        "id": p["id"], "name": p["name"], "status": p["desiredStatus"], "cost_per_hr": p["costPerHr"],
        "gpu": (p.get("machine") or {}).get("gpuDisplayName"),
        "uptime_min": (rt.get("uptimeInSeconds") or 0) // 60,
        "gpu_util_pct": gpu.get("gpuUtilPercent"), "gpu_mem_pct": gpu.get("memoryUtilPercent"),
        "cpu_pct": (rt.get("container") or {}).get("cpuPercent"),
        "tensorboard": f"https://{p['id']}-6006.proxy.runpod.net",
        "balance": round(d["myself"]["clientBalance"], 2),
    }, indent=2))


def cmd_terminate(args):
    if not args.yes:
        sys.exit("Terminating deletes the pod and its disk (and ends any Claude session on it). Re-run with --yes.")
    out = gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": args.pod_id})
    if out.get("errors"):
        sys.exit(f"terminate failed: {out['errors'][0].get('message')}")
    print(f"terminated {args.pod_id}")


def cmd_create(args):
    if args.idle_minutes < MIN_IDLE_MINUTES:
        sys.exit(f"--idle-minutes must be >= {MIN_IDLE_MINUTES}: every pod launched from this repo keeps the idle guard.")
    if not args.configs and not args.commands:
        sys.exit("Nothing to run: pass --configs and/or --commands.")
    branch = args.branch or current_branch()

    # The pod's start command: clone/update the repo onto the pod disk, start autorun.sh in the
    # background (idle guard, setup, Claude Code, jobs), then hand over to the image's own
    # start script (web terminal, SSH, Jupyter).
    boot = (
        "mkdir -p /workspace && cd /workspace && "
        f"(test -d vision-generative-models-survey || git clone -q -b {branch} {REPO_URL}) && "
        f"cd vision-generative-models-survey && git fetch -q origin {branch} && git checkout -q {branch} && git pull -q; "
        "nohup bash scripts/runpod/autorun.sh > /workspace/autorun.log 2>&1 & "
        "exec /start.sh"
    )
    env = {
        "AUTORUN_CONFIGS": ",".join(args.configs or []),
        "AUTORUN_COMMANDS": " ;; ".join(args.commands or []),
        "IDLE_MINUTES": str(args.idle_minutes),
        "MAX_HOURS": str(args.max_hours),
    }
    base = {
        "gpuCount": 1, "name": args.name, "imageName": IMAGE,
        "containerDiskInGb": args.disk_gb, "volumeInGb": args.disk_gb, "volumeMountPath": "/workspace",
        "minVcpuCount": 2, "minMemoryInGb": 8,
        # HTTP ports only: exposing a TCP port restricts the search to hosts with a public IP,
        # which ruled out every Community GPU when tried. Web terminal and SSH-over-proxy still work.
        "ports": "8888/http,6006/http",
        "startSsh": True,
        "dockerArgs": f"bash -c '{boot}'",
        "env": [{"key": k, "value": v} for k, v in env.items() if v],
    }
    gpus = DEFAULT_GPUS
    if args.gpu:
        gpus = [(c, g) for c, g in DEFAULT_GPUS if any(s.lower() in g.lower() for s in args.gpu)]
    if args.community_only:
        gpus = [(c, g) for c, g in gpus if c == "COMMUNITY"]
    if args.dry_run:
        print(json.dumps({"gpu_candidates": gpus, "input": base}, indent=2))
        return

    q = """mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) {
        id name costPerHr machine { gpuDisplayName location } } }"""
    for cloud, gpu in gpus:
        out = gql(q, {"input": dict(base, cloudType=cloud, gpuTypeId=gpu)})
        pod = (out.get("data") or {}).get("podFindAndDeployOnDemand")
        if pod:
            break
        print(f"  no {cloud.lower()} {gpu}: {(out.get('errors') or [{}])[0].get('message', '?')[:80]}")
    else:
        sys.exit("No GPU available right now from the candidate list. Try again later or widen --gpu.")

    pid = pod["id"]
    print(f"\nCreated pod {pid} ({pod['name']}): {pod['machine']['gpuDisplayName']}, "
          f"{cloud.lower()} cloud, ${pod['costPerHr']}/h")
    print(f"  cost guards: idle {args.idle_minutes} min, hard cap {args.max_hours} h "
          f"(<= ${pod['costPerHr'] * args.max_hours:.2f})")
    print(f"  jobs: commands={len(args.commands or [])} configs={args.configs or []}  (branch {branch})")
    print(f"  TensorBoard: https://{pid}-6006.proxy.runpod.net")
    for _ in range(12):  # wait for the container to come up
        rt = gql('query($id: String!) { pod(input: {podId: $id}) { runtime { uptimeInSeconds } } }', {"id": pid})
        if ((rt["data"]["pod"] or {}).get("runtime") or {}).get("uptimeInSeconds"):
            print("  container is up")
            break
        time.sleep(15)
    print(f"""
ONE MANUAL STEP (logins only the account owner can do; jobs wait for the Hugging Face one,
and the idle guard terminates the pod if nothing happens for {args.idle_minutes} min):
  Runpod console -> Pods -> {args.name} -> Connect -> Start Web Terminal, then run
    bash /workspace/vision-generative-models-survey/scripts/runpod/claude_remote.sh
  It logs in to Hugging Face and Claude and starts Claude Remote Control: the pod then shows
  up as session "runpod-genmodels" in claude.ai/code and the Claude app.""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(func=cmd_balance)
    sub.add_parser("list").set_defaults(func=cmd_list)
    s = sub.add_parser("status"); s.add_argument("pod_id"); s.set_defaults(func=cmd_status)
    t = sub.add_parser("terminate"); t.add_argument("pod_id"); t.add_argument("--yes", action="store_true")
    t.set_defaults(func=cmd_terminate)
    c = sub.add_parser("create", help="launch a pod that runs jobs, with the cost guards and Claude Remote Control")
    c.add_argument("--name", required=True)
    c.add_argument("--configs", nargs="*", help="configs to train side by side (pushed to the Hub)")
    c.add_argument("--commands", nargs="*", help="shell commands to run first, one after another")
    c.add_argument("--idle-minutes", type=int, default=30, help=f"idle guard (>= {MIN_IDLE_MINUTES}, always on)")
    c.add_argument("--max-hours", type=float, default=8, help="hard cap after boot")
    c.add_argument("--gpu", nargs="*", help="restrict to GPU names containing these strings, e.g. 4090 3090")
    c.add_argument("--community-only", action="store_true", help="never fall back to the pricier Secure Cloud")
    c.add_argument("--disk-gb", type=int, default=30)
    c.add_argument("--branch", help="git branch the pod checks out (default: the current one)")
    c.add_argument("--dry-run", action="store_true", help="print the request, create nothing")
    c.set_defaults(func=cmd_create)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
