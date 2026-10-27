"""Phase 5 Option B: failover with and without a KV checkpoint on the standby.

The first machine runs the gateway and replica 0; the standby (replica 1) runs on
a second machine. Both replicas run vLLM's OffloadingConnector writing KV blocks
to one shared directory (an NFS export of a RAM disk on the standby), or no
connector at all (the recompute baseline). In every trial:

  * the shared directory is emptied (--clear-cmd), so nothing carries over;
  * a fresh prompt of --prompt-len pseudo-random token ids is sent through the
    gateway (distinct per trial, so no trial can reuse another's blocks);
  * replica 0 is SIGKILLed once --kill-at output positions reached the client;
    the gateway resumes on the standby with prompt + delivered token ids, and
    the standby either loads the matching blocks from the shared directory or
    recomputes them;
  * the standby's offloading counters (/metrics, vllm:kv_offload_*) are read
    before and after, as evidence of whether blocks were loaded.

Raw records only; analysis/phase5b.py summarizes.

  python -m harness.run_ckpt_failover --run-type measured --label prompt \
    --server-cmd "vllm serve MODEL --port {port} ..." --model MODEL \
    --remote http://172.31.6.222:8000 --remote-desc "..." --prompt-len 16384 \
    --kill-at 128 --trials 5 --clear-cmd "rm -rf /srv/kv/*"
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import requests

from gateway.check import delivery
from gateway.client import stream_via_gateway
from harness.run_gateway_trial import wait_http
from harness.run_trial import OUT_DIRS, environment
from harness.server import ServerProcess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRIC = re.compile(r"^vllm:(kv_offload|prefix_cache|external_prefix_cache)")


def random_prompt(length, label):
    rng = random.Random(f"ckpt-{label}")
    return [rng.randrange(1000, 120000) for _ in range(length)]   # below the special tokens


def offload_metrics(url):
    try:
        text = requests.get(url + "/metrics", timeout=10).text
    except requests.RequestException as e:
        return {"error": str(e)}
    return {"lines": [ln for ln in text.splitlines() if METRIC.match(ln)]}


def run_cmd(cmd):
    if not cmd:
        return None
    t0 = time.monotonic()
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
    return {"cmd": cmd, "rc": r.returncode, "s": time.monotonic() - t0, "stderr": r.stderr[-500:]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--label", required=True, help="offload mode: none, prompt, all")
    p.add_argument("--server-cmd", required=True, help="replica 0 command, with {port}")
    p.add_argument("--model", required=True)
    p.add_argument("--remote", required=True, help="standby base URL")
    p.add_argument("--remote-desc", required=True)
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--kill-at", type=int, default=128)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--clear-cmd", default=None, help="shell command run before every trial")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gateway-port", type=int, default=9000)
    p.add_argument("--ready-timeout", type=float, default=900.0)
    a = p.parse_args()

    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_ckptfo_{a.label}"
              f"_L{a.prompt_len}_k{a.kill_at}")
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")
    url = f"http://127.0.0.1:{a.port}"
    server = ServerProcess(a.server_cmd.replace("{port}", str(a.port)), url,
                           os.path.join(out, "server_0.log"))
    gw_url = f"http://127.0.0.1:{a.gateway_port}"
    gw_cmd = [sys.executable, "-m", "gateway.server", "--backends", f"{url},{a.remote.rstrip('/')}",
              "--model", a.model, "--port", str(a.gateway_port),
              "--log-dir", os.path.join(out, "gateway_log"), "--mode", "ids",
              "--stall-timeout", "120", "--recovery-timeout", str(a.ready_timeout)]
    meta = {"run_id": run_id, "run_type": a.run_type, "label": a.label, "config": vars(a),
            "gateway_cmd": gw_cmd, "environment": env}
    gw_log = open(os.path.join(out, "gateway.log"), "a", encoding="utf-8")
    gw_proc = None
    try:
        server.start()
        meta["initial_startup_s"] = server.wait_ready(a.ready_timeout)
        wait_http(a.remote + "/v1/models", a.ready_timeout)
        try:
            meta["remote_version"] = requests.get(a.remote + "/version", timeout=5).json()
        except (requests.RequestException, ValueError):
            meta["remote_version"] = None
        gw_proc = subprocess.Popen(gw_cmd, cwd=REPO, stdout=gw_log, stderr=subprocess.STDOUT)
        wait_http(gw_url + "/health", 60)
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        for k in range(a.trials):
            if server.proc is None or server.proc.poll() is not None:
                server.start()
            server.wait_ready(a.ready_timeout)
            wait_http(a.remote + "/v1/models", a.ready_timeout)
            clear = run_cmd(a.clear_cmd)
            body = {"model": a.model, "prompt": random_prompt(a.prompt_len, f"{run_id}-{k}"),
                    "max_tokens": a.max_tokens, "temperature": 0.0, "seed": a.seed,
                    "ignore_eos": True}
            m_before = offload_metrics(a.remote)
            st, restarts = {}, []

            def hook(e):
                if st or e["seq"] + e["n"] < a.kill_at:
                    return
                if e["backend"] != 0:
                    st["not_killed"] = f"serving backend {e['backend']}"
                    return
                st["delivered_at_kill"] = e["seq"] + e["n"]
                st["t_kill"] = server.kill()

                def restart():
                    t0 = time.time()
                    server.start()
                    restarts.append({"t_start": t0, "ready_s": server.wait_ready(a.ready_timeout)})
                th = threading.Thread(target=restart, daemon=True)
                th.start()
                st["thread"] = th

            t0 = time.monotonic()
            res = stream_via_gateway(gw_url, body, on_event=hook)
            res["t_start_mono"] = t0
            if "thread" in st:
                st.pop("thread").join(a.ready_timeout)
            m_after = offload_metrics(a.remote)
            ev = res["events"]
            after = [e for e in ev if (e["attempt"] or 0) >= 1]
            first = ev[0]["t_mono"] - t0 if ev else None
            rec = {"trial": k, "run_id": run_id, "prompt_len": a.prompt_len, "kill_at": a.kill_at,
                   **st, "restarts": restarts, "clear": clear, "client": res,
                   "delivery": delivery(ev), "ttft_s": first,
                   "remote_metrics_before": m_before, "remote_metrics_after": m_after,
                   "recovery_latency_s": (after[0]["t_wall"] - st["t_kill"])
                   if after and "t_kill" in st else None}
            with open(os.path.join(out, f"trial_{k:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            d = rec["delivery"]
            print(f"trial {k}: {d['positions']} positions, dup {d['duplicated_positions']}, "
                  f"missing {d['missing_positions']}, TTFT {first and round(first, 2)} s, "
                  f"recovery {rec['recovery_latency_s'] and round(rec['recovery_latency_s'], 3)} s",
                  flush=True)
    finally:
        if gw_proc is not None:
            gw_proc.kill()
        server.stop()
        gw_log.close()
    print(f"raw logs in {out}")


if __name__ == "__main__":
    main()
