"""Phase 4 (RQ3), recompute side: how long does it take to rebuild a request's
KV cache by prefill, as a function of context length?

Recovery by recompute (Phase 1 client-side, Phase 3 gateway) resubmits the
prompt plus every delivered token, and the engine rebuilds the KV cache by
prefill before it can produce the next token. This script measures that cost
directly: for each context length L it sends REPS requests whose prompt is L
token ids and max_tokens=1, and records the time to the first (only) token.

Every prompt is a fresh pseudo-random sequence of token ids (seeded by L and the
repetition), so no request can reuse another's KV blocks; run the server with
prefix caching off as well (scripts/run_phase4.sh does). One unrecorded warm-up
request per length absorbs first-time effects at that shape.

Raw records only; analysis/phase4.py computes the curves.

Example:
  python -m harness.run_prefill --run-type measured --engine vllm --label noapc \
    --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 \
                  --max-model-len 33000 --no-enable-prefix-caching" \
    --model meta-llama/Llama-3.1-8B-Instruct --lengths 512,1024,2048,4096,8192,16384,32768
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

from harness.client import stream_completion
from harness.run_trial import OUT_DIRS, environment
from harness.server import ServerProcess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prompt_ids(length, rep, lo=1000, hi=120000):
    """Deterministic pseudo-random token ids; Llama 3.1's vocabulary is 128256."""
    rng = random.Random(f"prefill-{length}-{rep}")
    return [rng.randrange(lo, hi) for _ in range(length)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--server-cmd", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--ready-timeout", type=float, default=900.0)
    a = p.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_prefill_{a.engine}_{a.label}"
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")
    url = f"http://127.0.0.1:{a.port}"
    server = ServerProcess(a.server_cmd, url, os.path.join(out, "server.log"))
    meta = {"run_id": run_id, "run_type": a.run_type, "engine": a.engine, "label": a.label,
            "config": vars(a), "lengths": lengths, "environment": env}
    records = []
    try:
        server.start()
        meta["ready_s"] = server.wait_ready(timeout_s=a.ready_timeout)
        print(f"{run_id}: server ready after {meta['ready_s']:.0f} s")
        for L in lengths:
            stream_completion(url, a.model, prompt_ids(L, "warmup"), 1, 0.0, None)
            for rep in range(a.reps):
                t0 = time.monotonic()
                chunks, err = stream_completion(url, a.model, prompt_ids(L, rep), 1, 0.0, None)
                t1 = time.monotonic()
                rec = {"length": L, "rep": rep, "t_start_mono": t0, "t_end_mono": t1,
                       "t_first_mono": chunks[0]["t_mono"] if chunks else None,
                       "error": err}
                records.append(rec)
            ok = [r for r in records if r["length"] == L and not r["error"] and r["t_first_mono"]]
            if ok:
                ttft = sorted(r["t_first_mono"] - r["t_start_mono"] for r in ok)
                print(f"  L={L}: median TTFT {ttft[len(ttft) // 2]:.3f} s ({len(ok)}/{a.reps} ok)")
            else:
                print(f"  L={L}: all {a.reps} requests failed: {records[-1]['error']}")
    finally:
        server.stop()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(out, "requests.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
