"""Summarize Phase 3 (gateway) runs from raw logs. Every number is computed here.

    python analysis/gateway_check.py                    # experiments/
    python analysis/gateway_check.py --root smoke_tests

Kill runs (one row per run): trials, trials with exactly-once delivery,
duplicated and missing positions, trials whose final text equals the run's
failure-free reference, recovery latency (kill -> first recovered position at
the client) and gateway recovery attempts.

Overhead runs (one row per run): per-request time to first token, mean time
per output token after the first, and end-to-end time, direct to the engine
versus through the gateway, as medians over the pairs.

The delivery check is recomputed from the events the client recorded, not
taken from the trial file, so it cannot drift from gateway/check.py.
"""
import argparse
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from gateway.check import delivery, text_of  # noqa: E402


def med(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{fmt.format(statistics.median(xs))} ({fmt.format(min(xs))}-{fmt.format(max(xs))})"


def timing(t0, times):
    """ttft, mean inter-token time, end-to-end from a start time and arrival times."""
    if not times:
        return None, None, None
    ttft = times[0] - t0
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
    return ttft, tpot, times[-1] - t0


def kill_row(run_dir, meta):
    trials = []
    for f in sorted(glob.glob(os.path.join(run_dir, "trial_*.json"))):
        with open(f, encoding="utf-8") as fh:
            trials.append(json.load(fh))
    if not trials:
        return None
    ref = meta["reference"]["text"]
    ds = [delivery(t["client"]["events"]) for t in trials]
    texts = [text_of(t["client"]["events"]) for t in trials]
    c = meta["config"]
    return {
        "run_id": meta["run_id"], "engine": meta["engine"], "replicas": c["replicas"],
        "mode": c["mode"], "fsync": c["fsync"], "temperature": c["temperature"],
        "kill_at": c["kill_at"], "trials": len(trials),
        "exactly_once": sum(d["exactly_once"] for d in ds),
        "duplicated_positions": sum(d["duplicated_positions"] for d in ds),
        "missing_positions": sum(d["missing_positions"] for d in ds),
        "complete": sum(t["client"].get("finish_reason") in ("length", "stop") for t in trials),
        "matches_reference": sum(x == ref for x in texts),
        "reference_repeatable": meta.get("reference_repeat_identical"),
        "recovery_s": med([t.get("recovery_latency_s") for t in trials]),
        "attempts": med([(t["client"].get("final") or {}).get("attempts") for t in trials], "{:.0f}"),
        "git_commit": (meta["environment"].get("git_commit") or "")[:7],
    }


def overhead_row(run_dir, meta):
    path = os.path.join(run_dir, "overhead.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        pairs = json.load(fh)
    d = [timing(p["direct"]["t_start_mono"], [c["t_mono"] for c in p["direct"]["chunks"]])
         for p in pairs]
    g = [timing(p["gateway"]["t_start_mono"], [e["t_mono"] for e in p["gateway"]["events"]])
         for p in pairs]
    c = meta["config"]
    row = {"run_id": meta["run_id"], "engine": meta["engine"], "mode": c["mode"],
           "fsync": c["fsync"], "pairs": len(pairs)}
    for name, i, scale in (("ttft_ms", 0, 1000), ("tpot_ms", 1, 1000), ("e2e_s", 2, 1)):
        row[f"direct_{name}"] = med([x[i] * scale if x[i] is not None else None for x in d])
        row[f"gateway_{name}"] = med([x[i] * scale if x[i] is not None else None for x in g])
        diffs = [(y[i] - x[i]) * scale for x, y in zip(d, g) if x[i] is not None and y[i] is not None]
        row[f"diff_{name}"] = med(diffs)
    return row


def table(rows):
    if not rows:
        return []
    keys = list(rows[0])
    return (["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"))
    ap.add_argument("--out", default=None, help="write the Markdown summary here")
    a = ap.parse_args()
    kills, overheads = [], []
    for run_dir in sorted(glob.glob(os.path.join(a.root, "*_gw_*"))):
        mp = os.path.join(run_dir, "run.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        r = kill_row(run_dir, meta) if meta["config"].get("kill_at") is not None \
            else overhead_row(run_dir, meta)
        if r:
            (kills if "kill_at" in r else overheads).append(r)
    lines = ["## Gateway kill trials", "", *table(kills), "",
             "## Gateway overhead (failure-free; median (min-max) over pairs; diff = gateway - direct per pair)",
             "", *table(overheads)]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
