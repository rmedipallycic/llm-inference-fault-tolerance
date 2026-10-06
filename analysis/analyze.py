"""Compute client-visible anomaly statistics from raw trial logs.

Reads experiments/<run>/run.json and trial_*.json (or smoke_tests/ with
--smoke). Every number printed here is derived from those files only.

Per trial, comparing what the client saw against the failure-free reference:
  exact_match         client view == reference text
  duplicated_chars    length of the already-delivered prefix that the
                      recovery stream sent again (naive retry re-sends it)
  duplicated_fraction duplicated_chars / delivered_chars; 1.0 means the
                      client received everything it already had a second time
  recovery_equals_reference
                      the recovery stream alone reproduced the reference
                      (expected for naive retry when output is deterministic)
  first_divergence    first character index where client view and
                      reference differ (None if one is a prefix of the other)
  recovery_latency_s  kill -> first token delivered after recovery
                      (includes server restart time)
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def analyze_trial(trial, reference_text):
    delivered = "".join(c["text"] for c in trial["pre_kill_chunks"])
    recovered = "".join(c["text"] for c in trial["recovery_chunks"])
    view = trial["client_view_text"]
    dup = common_prefix_len(recovered, delivered)
    cp = common_prefix_len(view, reference_text)
    diverged = cp < min(len(view), len(reference_text))
    return {
        "trial": trial["trial"],
        "chunks_at_kill": trial.get("chunks_at_kill"),
        "delivered_chunks": len(trial["pre_kill_chunks"]),
        "recovered_chunks": len(trial["recovery_chunks"]),
        "exact_match": view == reference_text,
        "delivered_chars": len(delivered),
        "duplicated_chars": dup,
        "duplicated_fraction": (dup / len(delivered)) if delivered else None,
        "recovery_equals_reference": recovered == reference_text,
        "first_divergence": cp if diverged else None,
        "client_len": len(view),
        "reference_len": len(reference_text),
        "restart_s": trial["restart_s"],
        "recovery_latency_s": trial["recovery_latency_s"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="read smoke_tests/ instead of experiments/")
    p.add_argument("--csv", help="also write per-trial rows to this CSV path")
    a = p.parse_args()
    root = "smoke_tests" if a.smoke else "experiments"

    runs = sorted(glob.glob(os.path.join(root, "*", "run.json")))
    if not runs:
        sys.exit(f"no runs found under {root}/")

    rows = []
    for run_path in runs:
        run = json.load(open(run_path, encoding="utf-8"))
        ref = run["reference"]["text"]
        trials = sorted(glob.glob(os.path.join(os.path.dirname(run_path), "trial_*.json")))
        per = [analyze_trial(json.load(open(t, encoding="utf-8")), ref) for t in trials]
        for r in per:
            rows.append({"run_id": run["run_id"], "run_type": run["run_type"], **r})
        lat = [r["recovery_latency_s"] for r in per if r["recovery_latency_s"] is not None]
        print(f"\n{run['run_id']}  ({run['run_type']}, commit {str(run['environment']['git_commit'])[:8]})")
        print(f"  reference reproducible across two clean runs: {run['reference_repeat_identical']}")
        if not run["reference_repeat_identical"]:
            print("  WARNING: clean runs disagree, so divergence in this run is not interpretable")
        print(f"  trials: {len(per)}")
        if per:
            print(f"  exact match with reference: {sum(r['exact_match'] for r in per)}/{len(per)}")
            full = sum(r["duplicated_fraction"] == 1.0 for r in per)
            print(f"  re-sent the entire delivered prefix: {full}/{len(per)}")
            print(f"  duplicated fraction per trial: "
                  + ", ".join("n/a" if r["duplicated_fraction"] is None
                              else f"{r['duplicated_fraction']:.2f}" for r in per))
            print(f"  recovery stream identical to reference: "
                  f"{sum(r['recovery_equals_reference'] for r in per)}/{len(per)}")
            print(f"  trials with divergence:      {sum(r['first_divergence'] is not None for r in per)}/{len(per)}")
        if lat:
            print(f"  recovery latency (s): median {statistics.median(lat):.2f}, "
                  f"min {min(lat):.2f}, max {max(lat):.2f}")

    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-trial rows written to {a.csv}")


if __name__ == "__main__":
    main()
