"""Hardware costs for the Phase 5 simulator, read from committed raw logs.

Nothing measured is typed in here: every value is computed from experiments/
by the same rules as analysis/phase4.py and analysis/throughput.py, and the
run ids it came from are returned with it (docs/phase5-plan.md).
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
import gw_timing  # noqa: E402


def _runs(root, pattern, recfile):
    out = []
    for path in sorted(glob.glob(os.path.join(root, pattern, "run.json"))):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if run.get("run_type") != "measured":
            continue
        rp = os.path.join(os.path.dirname(path), recfile)
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                run["records"] = [json.loads(x) for x in f if x.strip()]
            out.append(run)
    return out


def load(root="experiments"):
    src = {}

    # Recompute cost R(L): median time to first token, pooled over the prefill runs
    pre = _runs(root, "*_prefill_*", "requests.jsonl")
    by_len = {}
    for r in pre:
        for x in r["records"]:
            if not x["error"]:
                by_len.setdefault(x["length"], []).append(
                    (x["t_first_mono"] or x["t_end_mono"]) - x["t_start_mono"])
    r_table = sorted((L, statistics.median(v)) for L, v in by_len.items())
    src["recompute"] = [r["run_id"] for r in pre]

    # KV bytes per token and GPU<->host bandwidth (largest size, median)
    cop = _runs(root, "*_kvcopy", "records.jsonl")[-1]
    big = max(x["length"] for x in cop["records"])
    rs = [x for x in cop["records"] if x["length"] == big]
    kv_bytes = cop["kv_bytes_per_token"]
    d2h_bw = rs[0]["bytes"] / statistics.median(x["d2h_s"] for x in rs)
    h2d_bw = rs[0]["bytes"] / statistics.median(x["h2d_s"] for x in rs)
    src["copy"] = [cop["run_id"]]

    # Network bandwidth to the standby: sizes-1 sweep at the largest size, median
    net = {}
    for r in _runs(root, "*_kvnet_*", "records.jsonl"):
        rs = [x for x in r["records"] if x["part"] == "sizes-1" and x["length"] == max(r["lengths"])]
        net[r["config"]["streams"]] = rs[0]["bytes"] / statistics.median(x["remote_recv_s"] for x in rs)
        src.setdefault("network", []).append(r["run_id"])

    # Time per output token: vLLM default, one request at a time (Phase 2 follow-up)
    tpots = []
    for r in _runs(root, "*_tput_vllm_default_*", "requests.jsonl"):
        for x in r["records"]:
            if x["level"] == 1 and not x["error"] and len(x["chunks"]) > 1:
                n = (x.get("usage") or {}).get("completion_tokens") or len(x["chunks"])
                tpots.append((x["chunks"][-1]["t_mono"] - x["chunks"][0]["t_mono"]) / (n - 1))
        src.setdefault("tpot", []).append(r["run_id"])
    tpot = statistics.median(tpots)

    params = {"r_table": r_table, "kv_bytes": kv_bytes, "d2h_bw": d2h_bw, "h2d_bw": h2d_bw,
              "net_bw": net, "tpot": tpot}

    # Failover detection and switch time D: Phase 3 failover recovery latency minus
    # the prefill of prompt + delivered tokens that it included (by R(n) below)
    ds = []
    for path in sorted(glob.glob(os.path.join(root, "*_gw_vllm_r1+1remote_*", "run.json"))):
        d = os.path.dirname(path)
        logs = sorted(glob.glob(os.path.join(d, "gateway_log", "*.jsonl")))
        with open(logs[0], encoding="utf-8") as f:
            n_prompt = len(json.loads(f.readline())["prompt_ids"])
        for tp in sorted(glob.glob(os.path.join(d, "trial_*.json"))):
            with open(tp, encoding="utf-8") as f:
                t = json.load(f)
            # recovery from the gateway's log, not the client (analysis/gw_timing.py:
            # the client saw events late until commit 64c39ec)
            rec = gw_timing.recovery_s(gw_timing.entries(d, t["client"].get("req_id")), t.get("t_kill"))
            if rec is not None:
                ds.append(rec - recompute_s(params, n_prompt + t["delivered_at_kill"]))
        src.setdefault("failover", []).append(os.path.basename(d))
    # can come out slightly negative if R(n) at this short length overestimates the
    # prefill; a negative switch time is not meaningful, so it is clamped at zero
    params["detect_s_raw"] = statistics.median(ds)
    params["detect_s"] = max(0.0, params["detect_s_raw"])
    params["sources"] = src
    return params


def recompute_s(p, n):
    """R(n): piecewise-linear in the measured points, through the origin below the
    smallest measured length. Not defined above the largest measured length."""
    tab = p["r_table"]
    if n <= 0:
        return 0.0
    if n <= tab[0][0]:
        return tab[0][1] * n / tab[0][0]
    for (l0, r0), (l1, r1) in zip(tab, tab[1:]):
        if n <= l1:
            return r0 + (r1 - r0) * (n - l0) / (l1 - l0)
    if n <= tab[-1][0] * 1.01:
        return tab[-1][1]
    raise ValueError(f"context {n} above the measured range ({tab[-1][0]})")
