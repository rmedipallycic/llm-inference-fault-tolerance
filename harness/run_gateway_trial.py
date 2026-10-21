"""Phase 3 (RQ2) harness: the exactly-once gateway on real engines.

Two kinds of run, both writing raw records only (analysis/gateway_check.py
computes every number afterwards):

  kill trials   stream a request through the gateway, SIGKILL the engine
                replica that is serving it once --kill-at positions have reached
                the client, restart that replica in the background, and record
                everything the client received. With --replicas 1 the gateway
                must wait for the restart; with --replicas 2 it fails over to
                the other replica.
  overhead      (--overhead-pairs N, no --kill-at) failure-free requests,
                alternating direct-to-engine and through-the-gateway, to
                measure what the gateway costs per request and per token.

The server command is a template: {port} and {gpu} are filled per replica,
e.g. "env CUDA_VISIBLE_DEVICES={gpu} vllm serve MODEL --port {port}".

Example (one A10G, vLLM, restart mode):
  python -m harness.run_gateway_trial --run-type measured --engine vllm \
    --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --port {port} --max-model-len 4096" \
    --model meta-llama/Llama-3.1-8B-Instruct --temperature 0.0 --seed 1234 \
    --max-tokens 256 --kill-at 128 --trials 10
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import requests

from gateway.check import delivery, first_divergence, text_of
from gateway.client import stream_via_gateway
from harness.client import join, stream_completion
from harness.run_trial import OUT_DIRS, PROMPTS_FILE, DEFAULT_PROMPT, environment
from harness.server import ServerProcess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wait_http(url, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return time.monotonic() - t0
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(url)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--engine", default="unspecified")
    p.add_argument("--server-cmd", required=True, help="template with {port} (and optionally {gpu})")
    p.add_argument("--model", required=True)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU index per replica, fills {gpu} (default 0,1,...)")
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--gateway-port", type=int, default=9000)
    p.add_argument("--mode", choices=["ids", "text"], default="ids")
    p.add_argument("--fsync", action="store_true")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--extra-body", default=None, help="JSON merged into every request")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--kill-at", type=int, default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--overhead-pairs", type=int, default=0)
    p.add_argument("--ready-timeout", type=float, default=900.0)
    p.add_argument("--stall-timeout", type=float, default=60.0)
    a = p.parse_args()

    if (a.kill_at is None) == (a.overhead_pairs == 0):
        p.error("give either --kill-at (kill trials) or --overhead-pairs (overhead run)")
    if a.kill_at is not None and not 1 <= a.kill_at < a.max_tokens:
        p.error("--kill-at must be between 1 and max-tokens - 1")
    if a.prompt_id is not None:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            prompts = {x["id"]: x["text"] for x in json.load(f)}
        a.prompt = prompts[a.prompt_id]
    extra = json.loads(a.extra_body) if a.extra_body else {}
    gpus = (a.gpus.split(",") if a.gpus else [str(i) for i in range(a.replicas)])

    kind = f"k{a.kill_at}" if a.kill_at is not None else "overhead"
    pid = f"_{a.prompt_id}" if a.prompt_id else ""
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_gw_{a.engine}{pid}"
              f"_r{a.replicas}_{a.mode}{'_fsync' if a.fsync else ''}_T{a.temperature}_{kind}")
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")

    ports = [a.base_port + i for i in range(a.replicas)]
    urls = [f"http://127.0.0.1:{pt}" for pt in ports]
    servers = [ServerProcess(a.server_cmd.format(port=pt, gpu=gpus[i]), urls[i],
                             os.path.join(out, f"server_{i}.log"))
               for i, pt in enumerate(ports)]
    gw_url = f"http://127.0.0.1:{a.gateway_port}"
    gw_cmd = [sys.executable, "-m", "gateway.server", "--backends", ",".join(urls),
              "--model", a.model, "--port", str(a.gateway_port),
              "--log-dir", os.path.join(out, "gateway_log"), "--mode", a.mode,
              "--stall-timeout", str(a.stall_timeout),
              "--recovery-timeout", str(a.ready_timeout)]
    if a.fsync:
        gw_cmd.append("--fsync")
    body = {"model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens,
            "temperature": a.temperature, **extra}
    if a.seed is not None:
        body["seed"] = a.seed
    meta = {"run_id": run_id, "run_type": a.run_type, "engine": a.engine, "config": vars(a),
            "gateway_cmd": gw_cmd, "environment": env, "backends": urls}
    gw_log = open(os.path.join(out, "gateway.log"), "a", encoding="utf-8")
    gw_proc = None
    try:
        for s in servers:
            s.start()
        meta["initial_startup_s"] = [s.wait_ready(a.ready_timeout) for s in servers]
        gw_proc = subprocess.Popen(gw_cmd, cwd=REPO, stdout=gw_log, stderr=subprocess.STDOUT)
        wait_http(gw_url + "/health", 60)

        def direct():
            t0 = time.monotonic()
            chunks, err = stream_completion(urls[0], a.model, a.prompt, a.max_tokens,
                                            a.temperature, a.seed, extra_body=extra or None)
            return {"t_start_mono": t0, "chunks": chunks, "error": err, "text": join(chunks)}

        def via_gateway(**kw):
            t0 = time.monotonic()
            res = stream_via_gateway(gw_url, body, **kw)
            res["t_start_mono"] = t0
            return res

        meta["reference"] = direct()
        meta["reference_repeat"] = direct()
        meta["reference_repeat_identical"] = meta["reference"]["text"] == meta["reference_repeat"]["text"]
        g = via_gateway()
        meta["gateway_reference"] = g
        meta["gateway_reference_text"] = text_of(g["events"])
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if a.overhead_pairs:
            pairs = []
            for k in range(a.overhead_pairs):
                first_direct = k % 2 == 0          # alternate order to cancel drift
                d1 = direct() if first_direct else via_gateway()
                d2 = via_gateway() if first_direct else direct()
                d, gw = (d1, d2) if first_direct else (d2, d1)
                pairs.append({"pair": k, "direct": d, "gateway": gw,
                              "gateway_first": not first_direct})
                print(f"pair {k}: direct {len(d['chunks'])} chunks, gateway {len(gw['events'])} events")
            with open(os.path.join(out, "overhead.json"), "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2)
            return

        for k in range(a.trials):
            for s in servers:                     # every replica up before a trial starts
                if s.proc is None or s.proc.poll() is not None:
                    s.start()
                s.wait_ready(a.ready_timeout)
            st = {}
            restarts = []

            def hook(e):
                if st or e["seq"] + e["n"] < a.kill_at:
                    return
                victim = servers[e["backend"]]
                st["killed_backend"] = e["backend"]
                st["delivered_at_kill"] = e["seq"] + e["n"]
                st["t_kill"] = victim.kill()

                def restart():
                    t0 = time.time()
                    victim.start()
                    restarts.append({"backend": st["killed_backend"], "t_start": t0,
                                     "ready_s": victim.wait_ready(a.ready_timeout)})
                th = threading.Thread(target=restart, daemon=True)
                th.start()
                st["thread"] = th

            res = via_gateway(on_event=hook)
            if "thread" in st:
                st.pop("thread").join(a.ready_timeout)
            ev = res["events"]
            after = [e for e in ev if (e["attempt"] or 0) >= 1]
            text = text_of(ev)
            rec = {"trial": k, "run_id": run_id, "kill_at": a.kill_at, **st,
                   "restarts": restarts, "client": res,
                   "delivery": delivery(ev),
                   "client_text": text,
                   "matches_reference": text == meta["reference"]["text"],
                   "first_divergence_from_reference": first_divergence(text, meta["reference"]["text"]),
                   "recovery_latency_s": (after[0]["t_wall"] - st["t_kill"]) if after and "t_kill" in st else None}
            with open(os.path.join(out, f"trial_{k:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            d = rec["delivery"]
            print(f"trial {k}: {d['positions']} positions, dup {d['duplicated_positions']}, "
                  f"missing {d['missing_positions']}, matches ref {rec['matches_reference']}, "
                  f"recovery {rec['recovery_latency_s']}")
    finally:
        if gw_proc is not None:
            gw_proc.kill()
        for s in servers:
            s.stop()
        gw_log.close()
    print(f"raw logs in {out}")


if __name__ == "__main__":
    main()
