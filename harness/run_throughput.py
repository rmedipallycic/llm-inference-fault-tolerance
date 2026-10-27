"""Phase 2 follow-up: what the fixes for the anomalies cost in throughput.

Phase 1 found two fixes: each engine's deterministic mode (vLLM batch
invariance, SGLang deterministic inference) and, on vLLM, disabling automatic
prefix caching. This script measures their failure-free cost under concurrent
load, which Phase 1 never tested (it issued requests one at a time).

One run = one server configuration. The script starts the server, sends a few
unrecorded warm-up requests, then for each concurrency level C sends N
requests with C in flight at a time and records every streamed chunk. Requests
cycle through the six prompts in harness/prompts.json and use ignore_eos, so
every request generates exactly --max-tokens tokens. Raw records only;
analysis/throughput.py computes every number afterwards, including whether
the temperature-0 output for each prompt stayed the same across requests and
concurrency levels.

Example (one A10G, vLLM default):
  python -m harness.run_throughput --run-type measured --engine vllm --label default \
    --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-model-len 4096" \
    --model meta-llama/Llama-3.1-8B-Instruct --concurrency 1,4,16,32
"""
import argparse
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

from harness.run_trial import OUT_DIRS, PROMPTS_FILE, environment
from harness.server import ServerProcess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one_request(base_url, body, timeout_s=900):
    """Stream one completion; returns a raw record with chunk timestamps."""
    rec = {"t_start_mono": time.monotonic(), "t_start_wall": time.time(),
           "chunks": [], "usage": None, "error": None, "finish_reason": None}
    try:
        with requests.post(f"{base_url}/v1/completions", json=body, stream=True,
                           timeout=timeout_s) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if payload.get("usage"):
                    rec["usage"] = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    rec["finish_reason"] = choices[0]["finish_reason"]
                text = choices[0].get("text", "")
                if text:
                    rec["chunks"].append({"text": text, "t_mono": time.monotonic()})
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_end_mono"] = time.monotonic()
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--label", required=True,
                   help="server configuration, e.g. default, batch-invariant, no-apc, deterministic")
    p.add_argument("--server-cmd", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--concurrency", default="1,4,16,32")
    p.add_argument("--requests-per-level", default=None,
                   help="comma-separated, one per level (default: max(12, 4*C))")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--extra-body", default=None, help="JSON merged into every request")
    p.add_argument("--ready-timeout", type=float, default=900.0)
    p.add_argument("--random-prompt-len", type=int, default=0,
                   help="Phase 5 Option B: every request gets its own prompt of this many "
                        "pseudo-random token ids instead of harness/prompts.json")
    p.add_argument("--pre-start-cmd", default=None,
                   help="shell command run before the server starts")
    p.add_argument("--clear-cmd", default=None,
                   help="shell command run before the warm-up and before every level "
                        "(Phase 5 Option B: empty the shared KV directory)")
    a = p.parse_args()

    levels = [int(x) for x in a.concurrency.split(",")]
    if a.requests_per_level:
        counts = [int(x) for x in a.requests_per_level.split(",")]
        if len(counts) != len(levels):
            p.error("--requests-per-level needs one count per concurrency level")
    else:
        counts = [max(12, 4 * c) for c in levels]
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        prompts = json.load(f)
    extra = json.loads(a.extra_body) if a.extra_body else {}

    kind = f"ckptovh_L{a.random_prompt_len}" if a.random_prompt_len else "tput"
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
              f"_{kind}_{a.engine}_{a.label}_T{a.temperature}")
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")
    url = f"http://127.0.0.1:{a.port}"
    server = ServerProcess(a.server_cmd, url, os.path.join(out, "server.log"),
                           pre_start=a.pre_start_cmd)

    def random_ids(tag):
        rng = random.Random(f"ovh-{a.random_prompt_len}-{tag}")
        return [rng.randrange(1000, 120000) for _ in range(a.random_prompt_len)]

    def clear():
        if a.clear_cmd:
            subprocess.run(["bash", "-c", a.clear_cmd], timeout=600)

    def body_for(prompt_text):
        b = {"model": a.model, "prompt": prompt_text, "max_tokens": a.max_tokens,
             "temperature": a.temperature, "stream": True, "ignore_eos": True,
             "stream_options": {"include_usage": True}, **extra}
        if a.seed is not None:
            b["seed"] = a.seed
        return b

    meta = {"run_id": run_id, "run_type": a.run_type, "engine": a.engine, "label": a.label,
            "config": vars(a), "levels": levels, "counts": counts,
            "prompts": prompts, "environment": env}
    try:
        server.start()
        meta["ready_s"] = server.wait_ready(timeout_s=a.ready_timeout)
        print(f"{run_id}: server ready after {meta['ready_s']:.0f} s")
        clear()
        for i in range(a.warmup):           # not recorded: first-request effects
            one_request(url, body_for(random_ids(f"w{i}") if a.random_prompt_len
                                      else prompts[i % len(prompts)]["text"]))
        records = []
        lock = threading.Lock()
        seq = [0]

        def job(level, k):
            if a.random_prompt_len:
                rec = one_request(url, body_for(random_ids(f"{level}-{k}")))
                rec.update({"level": level, "k": k, "prompt_id": f"random-{level}-{k}"})
            else:
                pr = prompts[k % len(prompts)]
                rec = one_request(url, body_for(pr["text"]))
                rec.update({"level": level, "k": k, "prompt_id": pr["id"]})
            with lock:
                rec["done_order"] = seq[0]
                seq[0] += 1
                records.append(rec)

        meta["level_windows"] = []
        for level, n in zip(levels, counts):
            clear()
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=level) as ex:
                list(ex.map(lambda k: job(level, k), range(n)))
            t1 = time.monotonic()
            meta["level_windows"].append({"level": level, "requests": n,
                                          "t_start_mono": t0, "t_end_mono": t1})
            errs = sum(1 for r in records if r["level"] == level and r["error"])
            print(f"  C={level}: {n} requests in {t1 - t0:.1f} s, errors {errs}")
        with open(os.path.join(out, "requests.jsonl"), "w", encoding="utf-8") as f:
            for r in sorted(records, key=lambda r: (r["level"], r["k"])):
                f.write(json.dumps(r) + "\n")
    finally:
        server.stop()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
