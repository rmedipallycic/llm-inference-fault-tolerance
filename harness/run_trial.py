"""Phase 1 fault-injection harness.

One run = start the server, record failure-free reference outputs, then
for each trial: stream a request, SIGKILL the server at a chosen chunk,
restart it, apply a client-side recovery strategy, and write the full
raw record to JSON. Nothing here computes or invents results; analysis
reads these raw files later.

Recovery strategies (client side, unmodified engine):
  naive_retry  resubmit the original prompt, append whatever comes back
  recompute    resubmit prompt + already-delivered text, ask for the rest

Example (AWS GPU instance, vLLM):
  python -m harness.run_trial --run-type measured \
    --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --strategy naive_retry --temperature 0.8 --seed 1234 \
    --max-tokens 256 --kill-at 128 --trials 10
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from harness.client import join, stream_completion
from harness.server import ServerProcess

DEFAULT_PROMPT = (
    "Explain, in a few paragraphs, how a write-ahead log lets a database "
    "recover after a crash."
)
OUT_DIRS = {"smoke": "smoke_tests", "measured": "experiments"}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def package_versions():
    """Versions of the serving stack, if installed in this Python environment."""
    from importlib import metadata
    out = {}
    for pkg in ("vllm", "sglang", "torch", "transformers", "requests"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def gpu_inventory():
    """One entry per GPU from nvidia-smi; empty list if no NVIDIA GPU."""
    raw = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
              "--format=csv,noheader"])
    gpus = []
    for line in (raw or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4:
            gpus.append({"index": parts[0], "name": parts[1],
                         "memory_total": parts[2], "driver": parts[3]})
    return gpus


def environment():
    gpus = gpu_inventory()
    return {
        "packages": package_versions(),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        # Only code counts: new logs in experiments/ must not mark a run dirty.
        "git_dirty": bool(sh(["git", "status", "--porcelain", "--",
                              "harness", "analysis", "scripts", "requirements.txt"])),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True,
                   help="smoke -> smoke_tests/ (never reported); measured -> experiments/")
    p.add_argument("--server-cmd", required=True, help="command that starts the server")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="model name the server expects")
    p.add_argument("--engine", default="unspecified", help="label: vllm, sglang, llamacpp")
    p.add_argument("--strategy", choices=["naive_retry", "recompute"], required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--kill-at", type=int, required=True,
                   help="kill the server after this many chunks have arrived")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--ready-timeout", type=float, default=900.0)
    a = p.parse_args()

    if a.kill_at < 1 or a.kill_at >= a.max_tokens:
        p.error("--kill-at must be between 1 and max-tokens - 1")

    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
              f"_{a.engine}_{a.strategy}_T{a.temperature}_k{a.kill_at}")
    out = os.path.join(OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")

    server = ServerProcess(a.server_cmd, a.base_url, os.path.join(out, "server.log"))
    gen = dict(base_url=a.base_url, model=a.model, temperature=a.temperature, seed=a.seed)

    run_meta = {"run_id": run_id, "run_type": a.run_type, "engine": a.engine,
                "config": vars(a), "environment": env}

    try:
        server.start()
        run_meta["initial_startup_s"] = server.wait_ready(a.ready_timeout)

        # Two failure-free references: is the output even reproducible?
        refs = []
        for _ in range(2):
            chunks, err = stream_completion(prompt=a.prompt, max_tokens=a.max_tokens, **gen)
            refs.append({"chunks": chunks, "error": err, "text": join(chunks)})
        run_meta["reference"] = refs[0]
        run_meta["reference_repeat"] = refs[1]
        run_meta["reference_repeat_identical"] = refs[0]["text"] == refs[1]["text"]
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        for k in range(a.trials):
            rec = {"trial": k, "run_id": run_id, "strategy": a.strategy,
                   "kill_at": a.kill_at}
            kill_info = {}

            def on_chunk(i, _text):
                if i + 1 == a.kill_at and not kill_info:
                    kill_info["t_kill"] = server.kill()
                    kill_info["chunks_at_kill"] = i + 1

            pre, err = stream_completion(prompt=a.prompt, max_tokens=a.max_tokens,
                                         on_chunk=on_chunk, **gen)
            if "t_kill" not in kill_info:  # finished before reaching kill point
                kill_info["t_kill"] = server.kill()
                rec["note"] = "stream ended before kill point; killed afterwards"
            rec["t_kill"] = kill_info["t_kill"]
            rec["chunks_at_kill"] = kill_info.get("chunks_at_kill")
            # chunks the server had already sent before dying still reach the client
            rec["in_flight_chunks_after_kill"] = (len(pre) - kill_info["chunks_at_kill"]
                                                  if kill_info.get("chunks_at_kill") else None)
            rec["pre_kill_chunks"] = pre
            rec["pre_kill_stream_end"] = err
            delivered = join(pre)

            server.start()
            rec["restart_s"] = server.wait_ready(a.ready_timeout)

            if a.strategy == "naive_retry":
                rec_prompt, rec_max = a.prompt, a.max_tokens
            else:
                rec_prompt = a.prompt + delivered
                rec_max = max(1, a.max_tokens - len(pre))
            rec["recovery_request"] = {"prompt_suffix_chars": len(rec_prompt) - len(a.prompt),
                                       "max_tokens": rec_max}
            post, err2 = stream_completion(prompt=rec_prompt, max_tokens=rec_max, **gen)
            rec["recovery_chunks"] = post
            rec["recovery_stream_end"] = err2
            rec["recovery_latency_s"] = (post[0]["t_wall"] - rec["t_kill"]) if post else None
            rec["client_view_text"] = delivered + join(post)

            path = os.path.join(out, f"trial_{k:03d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            print(f"trial {k}: delivered {len(pre)} chunks, recovered {len(post)}, "
                  f"restart {rec['restart_s']:.1f}s -> {path}")
    finally:
        server.stop()
    print(f"raw logs in {out}")


if __name__ == "__main__":
    main()
