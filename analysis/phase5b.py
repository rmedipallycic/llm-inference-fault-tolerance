"""Phase 5 Option B summary, from raw records (measured, not simulated).

  python analysis/phase5b.py --out experiments/phase5b_summary.md

Overhead runs (experiments/*_ckptovh_*, harness/run_throughput.py with
--random-prompt-len): per offload mode and concurrency, median time to first
token, median time per output token, output tokens/s, and each relative to the
mode "none" (mean over repetitions).

Failover runs (experiments/*_ckptfo_*, harness/run_ckpt_failover.py): per
offload mode and prompt length, delivery (every position exactly once), recovery
latency (kill to the next position at the client) and the KV bytes the standby
loaded from the shared directory during the trial (vllm:kv_offload_load_bytes,
from its /metrics before and after).

"Echo": trials in which the text of the first entry after the resume repeated
the text already delivered (vLLM returned the decoded prompt and delivered text
as that token's text; see docs/phase5b.md). Positions can be exactly-once while
the concatenated text is not. "Text ok": records whose concatenated client text
equals the standby's /detokenize of the delivered ids (recorded from the
revised harness on; "-" for earlier runs). With --concurrency C each trial has C
records, one per request, and the counts are over records.
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw_timing  # noqa: E402

MODES = ("none", "prompt", "all")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, s="{:.3f}"):
    return s.format(x) if x is not None else "-"


def counter(m, name):
    """Sum of a Prometheus counter's samples (all label sets) in captured lines."""
    tot, seen = 0.0, False
    for ln in (m or {}).get("lines", []):
        head, _, val = ln.rpartition(" ")
        metric = head.split("{")[0]
        if metric in (name, name + "_total"):
            try:
                tot += float(val)
                seen = True
            except ValueError:
                pass
    return tot if seen else None


def load_runs(root, pattern):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, pattern, "run.json"))):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if run.get("run_type") == "measured":
            run["_dir"] = os.path.dirname(path)
            runs.append(run)
    return runs


def overhead(root):
    rows = []
    for run in load_runs(root, "*_ckptovh_*"):
        rp = os.path.join(run["_dir"], "requests.jsonl")
        if not os.path.exists(rp):
            continue
        with open(rp, encoding="utf-8") as f:
            recs = [json.loads(x) for x in f if x.strip()]
        mode = run["label"].replace("offload-", "")
        for w in run["level_windows"]:
            ok = [r for r in recs if r["level"] == w["level"] and not r["error"] and len(r["chunks"]) > 1]
            ntok = [(r.get("usage") or {}).get("completion_tokens") or len(r["chunks"]) for r in ok]
            rows.append({"run_id": run["run_id"], "mode": mode, "C": w["level"],
                         "requests": len([r for r in recs if r["level"] == w["level"]]),
                         "errors": len([r for r in recs if r["level"] == w["level"] and r["error"]]),
                         "ttft_ms": med([(r["chunks"][0]["t_mono"] - r["t_start_mono"]) * 1000 for r in ok]),
                         "tpot_ms": med([(r["chunks"][-1]["t_mono"] - r["chunks"][0]["t_mono"]) / (n - 1) * 1000
                                         for r, n in zip(ok, ntok)]),
                         "tps": sum(ntok) / (w["t_end_mono"] - w["t_start_mono"])})
    return rows


def echoed(trial):
    ev = trial["client"]["events"]
    i = next((k for k, e in enumerate(ev) if (e["attempt"] or 0) >= 1), None)
    if i is None:
        return False
    before = "".join(e["text"] for e in ev[:i])
    t = ev[i]["text"] or ""
    # the delivered text appears again inside a much longer text (decoded prompt +
    # delivered text + the new token's text)
    return bool(before) and len(t) > len(before) + 64 and before in t


def text_ok(d, trials):
    """Per record: client text equals the reference decoding. From text_recheck.json
    (analysis/recheck_text.py, special tokens by id) when present, else from the
    harness's own check."""
    p = os.path.join(d, "text_recheck.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)["records"]
        return [rec[t["_file"]]["equal"] for t in trials if t["_file"] in rec]
    return [t["text_check"]["equal"] for t in trials if t.get("text_check")]


def gw_source(d, t, resumed):
    for e in gw_timing.tokens(gw_timing.entries(d, t["client"].get("req_id"))):
        if bool(e.get("attempt")) == resumed and e.get("text_source"):
            return e["text_source"]
    return ""


def gw_rec(d, t):
    x = gw_timing.recovery_s(gw_timing.entries(d, t["client"].get("req_id")), t.get("t_kill"))
    return x if x is not None else t.get("recovery_latency_s")


def gw_ttft(d, t):
    x = gw_timing.ttft_s(gw_timing.entries(d, t["client"].get("req_id")), t["client"])
    return x if x is not None else t.get("ttft_s")


def failover(root):
    rows = []
    for run in load_runs(root, "*_ckptfo_*"):
        trials = []
        for tp in sorted(glob.glob(os.path.join(run["_dir"], "trial_*.json"))):
            with open(tp, encoding="utf-8") as f:
                trials.append(json.load(f))
            trials[-1]["_file"] = os.path.basename(tp)
        if not trials:
            continue
        loaded = []
        for t in trials:
            b, a = (counter(t.get("remote_metrics_before"), "vllm:kv_offload_load_bytes"),
                    counter(t.get("remote_metrics_after"), "vllm:kv_offload_load_bytes"))
            loaded.append(a - b if a is not None and b is not None else None)
        eo = sum(1 for t in trials if t["delivery"]["exactly_once"]
                 and t["client"].get("finish_reason") in ("length", "stop"))
        rows.append({"run_id": run["run_id"], "mode": run["label"], "L": run["config"]["prompt_len"],
                     "trials": len(trials), "killed": sum(1 for t in trials if "t_kill" in t),
                     "exactly_once": eo,
                     "echo": sum(1 for t in trials if echoed(t)),
                     "C": run["config"].get("concurrency", 1),
                     # records where the gateway removed text the engine repeated after a
                     # resume (text_source in the gateway log; revised gateway only)
                     "repeat_removed": sum(1 for t in trials if gw_source(run["_dir"], t, resumed=True)
                                           .startswith("detokenized")),
                     "text_ok": text_ok(run["_dir"], trials),
                     "commit": run["environment"].get("git_commit", "")[:7],
                     # recovery and first token from the gateway's log (analysis/gw_timing.py);
                     # client-side figures kept for comparison
                     "rec": [gw_rec(run["_dir"], t) for t in trials],
                     "ttft": [gw_ttft(run["_dir"], t) for t in trials],
                     "rec_client": [t["recovery_latency_s"] for t in trials],
                     "loaded_mb": [x / 1e6 if x is not None else None for x in loaded]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    L = ["## Phase 5 Option B: KV checkpoint inside vLLM (measured)", "",
         "Generated by `analysis/phase5b.py` from `experiments/*_ckptovh_*` and `*_ckptfo_*`.", ""]

    ov = overhead(a.root)
    if ov:
        base = {}
        for r in ov:
            if r["mode"] == "none":
                base.setdefault(r["C"], []).append(r)
        L += ["### Decoding cost of offloading (prompt length in the run id; 256 output tokens)", "",
              "| run_id | mode | C | requests | errors | TTFT ms | TPOT ms | TPOT vs none | tok/s | tok/s vs none |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ov:
            b = base.get(r["C"])
            bt = statistics.mean(x["tpot_ms"] for x in b) if b else None
            bs = statistics.mean(x["tps"] for x in b) if b else None
            L.append(f"| {r['run_id']} | {r['mode']} | {r['C']} | {r['requests']} | {r['errors']} "
                     f"| {fmt(r['ttft_ms'], '{:.0f}')} | {fmt(r['tpot_ms'], '{:.2f}')} "
                     f"| {fmt(r['tpot_ms'] / bt if bt and r['tpot_ms'] else None, '{:.3f}x')} "
                     f"| {r['tps']:.1f} | {fmt(r['tps'] / bs if bs else None, '{:.3f}x')} |")
        L.append("")

    fo = failover(a.root)
    if fo:
        L += ["### Failover to the standby (kill position in the run id)", "",
              "| run_id | mode | prompt tokens | trials | killed | exactly-once | recovery s, median (min-max) "
              "| TTFT s, median | KV the standby loaded, MB, median (vllm:kv_offload_load_bytes delta) | echo "
              "| C | text ok | commit |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(fo, key=lambda r: (r["C"], r["L"], MODES.index(r["mode"]) if r["mode"] in MODES else 9,
                                           r["run_id"])):
            rec = [x for x in r["rec"] if x is not None]
            tok = f"{sum(1 for x in r['text_ok'] if x)}/{len(r['text_ok'])}" if r["text_ok"] else "-"
            rr = f"{med(rec):.2f} ({min(rec):.2f}-{max(rec):.2f})" if rec else "-"
            L.append(f"| {r['run_id']} | {r['mode']} | {r['L']} | {r['trials']} | {r['killed']} "
                     f"| {r['exactly_once']}/{r['trials']} | {rr} | {fmt(med(r['ttft']), '{:.2f}')} "
                     f"| {fmt(med(r['loaded_mb']), '{:.0f}')} | {r['echo']}/{r['trials']} | {r['C']} "
                     f"| {tok} "
                     f"| {r['commit']} |")
        L.append("")
    text = "\n".join(L) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
