"""Generate the paper's LaTeX tables directly from raw measured logs.

Reads experiments/<run>/run.json and trial_*.json and writes:

  paper/tables/repro.tex      failure-free reproducibility: does the first
                              request on a fresh server (G) equal the second (G')?
                              Also how many distinct G and G' texts occurred across
                              the runs of one configuration and temperature.
  paper/tables/anomalies.tex  one row per (configuration, strategy, T, kill point)
  paper/tables/latency.tex    restart, time to first token after restart, and
                              kill -> first recovered token
  paper/tables/env.tex        engine/library versions and GPU per configuration
  paper/tables/provenance.tex runs, trials, commits
  paper/tables/gw_*.tex       Phase 3 gateway tables (see analysis/gw_tables.py)
  paper/tables/{tput,restore,sim,p5b_*,v3_macros}.tex  Phases 2, 4, 5 (analysis/v3_tables.py)

No number is typed by hand: re-running this script on the committed logs
reproduces every table.

Why two references. Each run records two failure-free outputs for the same
request: G, the first request after the server starts, and G', the second.
In the vLLM runs these differ (deterministically), so "the" failure-free
output is not unique and every comparison is made against both.

Per trial, with D = text delivered before the kill and R = recovery text:
  final text F  naive retry: F = R (the retry is a complete answer)
                recompute:   F = D + R
  contradicts   naive retry only: R does not start with D, i.e. the replay
                disagrees with text the client has already shown. (Recompute
                cannot contradict D by construction.)
  F = G, F = G' exact equality with either failure-free reference

    python analysis/make_tables.py            # reads experiments/
    python analysis/make_tables.py --root X   # any directory of runs
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

CONFIG_SHORT = {"vllm": "vLLM", "vllm-noapc": "vLLM, no APC",
                "vllm-bi": "vLLM, batch-inv.",
                "sglang": "SGLang", "sglang-noradix": "SGLang, no radix",
                "sglang-seed": "SGLang, fixed seed", "sglang-det": "SGLang, determ."}
STRATEGY_NAME = {"naive_retry": "naive", "recompute": "recomp.",
                 "recompute_ids": "recomp. (ids)"}
CONFIG_ORDER = ["vllm", "vllm-noapc", "vllm-bi",
                "sglang", "sglang-noradix", "sglang-seed", "sglang-det", "llamacpp"]


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def join(chunks):
    return "".join(c["text"] for c in chunks)


def lcp(a, b):
    n, i = min(len(a), len(b)), 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def frac(k, n):
    return f"{k}/{n}" if n else "--"


def med_range(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    f = lambda v: fmt.format(v)  # noqa: E731
    return f"{f(statistics.median(xs))} ({f(min(xs))}--{f(max(xs))})"


def order_key(label):
    return (CONFIG_ORDER.index(label) if label in CONFIG_ORDER else 99, label)


def trial_metrics(t, G, G2):
    D, R = join(t["pre_kill_chunks"]), join(t["recovery_chunks"])
    rq = t.get("recovery_request") or {}
    naive = t["strategy"] == "naive_retry"
    F = R if naive else D + R
    return {
        "delivered_chars": len(D),
        "contradicts": (not R.startswith(D)) if naive else None,
        "contradict_at": lcp(R, D) if naive and not R.startswith(D) else None,
        "F_eq_G": F == G,
        "F_eq_G2": F == G2,
        "F_lcp_G": lcp(F, G),
        "F_lcp_G2": lcp(F, G2),
        "restart_s": t["restart_s"],
        "recovery_latency_s": t["recovery_latency_s"],
        "ttft_after_restart_s": (t["recovery_latency_s"] - t["restart_s"]
                                 if t["recovery_latency_s"] is not None else None),
        "in_flight": t.get("in_flight_chunks_after_kill"),
        "retok_matches": rq.get("retokenization_matches"),
    }


def load_runs(root):
    runs = []
    for run_path in sorted(glob.glob(os.path.join(root, "*", "run.json"))):
        with open(run_path, encoding="utf-8") as f:
            run = json.load(f)
        if run.get("run_type") != "measured":
            continue  # smoke runs are never reported
        if "strategy" not in run.get("config", {}):
            continue  # Phase 3 gateway runs: summarized by analysis/gateway_check.py
        G, G2 = run["reference"]["text"], run["reference_repeat"]["text"]
        trials = []
        for p in sorted(glob.glob(os.path.join(os.path.dirname(run_path), "trial_*.json"))):
            with open(p, encoding="utf-8") as f:
                trials.append(trial_metrics(json.load(f), G, G2))
        runs.append((run, trials))
    return runs


def header(f):
    f.write("% generated by analysis/make_tables.py; do not edit by hand\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--out", default=os.path.join("paper", "tables"))
    a = p.parse_args()

    runs = load_runs(a.root)
    if not runs:
        sys.exit(f"no measured runs under {a.root}/; nothing written")
    os.makedirs(a.out, exist_ok=True)

    repro = defaultdict(list)          # (label, T) -> [(identical, lcp, lenG, lenG2, G, G2)]
    cells = defaultdict(list)          # (label, strategy, T, k) -> trial metrics
    lat = defaultdict(list)            # (label, strategy) -> trial metrics
    env_by_label, commits = {}, set()
    n_trials = 0
    prompt_cells = defaultdict(list)   # (label, prompt_id, strategy) -> trial metrics
    prompt_repro = defaultdict(list)   # (label, prompt_id) -> [G == G']
    for run, trials in runs:
        cfg, label = run["config"], run["engine"]
        if cfg.get("prompt_id"):       # extra prompts get their own table
            pid = cfg["prompt_id"]
            prompt_cells[(label, pid, cfg["strategy"])].extend(trials)
            prompt_repro[(label, pid)].append(run["reference"]["text"] == run["reference_repeat"]["text"])
            commits.add(str(run["environment"].get("git_commit"))[:7])
            n_trials += len(trials)
            env_by_label.setdefault(label, (run["environment"], cfg["server_cmd"]))
            continue
        T = float(cfg["temperature"])
        G, G2 = run["reference"]["text"], run["reference_repeat"]["text"]
        # token-id runs request logprobs on every call, which is a different request
        # path; they are reported in retok.tex, not in the reproducibility table
        if cfg["strategy"] != "recompute_ids":
            repro[(label, T)].append((G == G2, lcp(G, G2), len(G), len(G2), G, G2))
        cells[(label, cfg["strategy"], T, int(cfg["kill_at"]))].extend(trials)
        lat[(label, cfg["strategy"])].extend(trials)
        env_by_label.setdefault(label, (run["environment"], cfg["server_cmd"]))
        commits.add(str(run["environment"].get("git_commit"))[:7])
        n_trials += len(trials)

    # reproducibility
    with open(os.path.join(a.out, "repro.tex"), "w", encoding="utf-8") as f:
        header(f)
        for (label, T) in sorted(repro, key=lambda k: (order_key(k[0]), k[1])):
            rs = repro[(label, T)]
            same = sum(r[0] for r in rs)
            lcps = sorted({r[1] for r in rs})
            lcp_s = str(lcps[0]) if len(lcps) == 1 else f"{lcps[0]}--{lcps[-1]}"
            nG = len({r[4] for r in rs})
            nG2 = len({r[5] for r in rs})
            f.write(f"{CONFIG_SHORT.get(label, tex(label))} & {T:g} & {frac(same, len(rs))} & "
                    f"{lcp_s} & {nG} & {nG2} \\\\\n")

    # anomalies
    rows = []
    for key in sorted(cells, key=lambda k: (order_key(k[0]), k[1], k[2], k[3])):
        label, strat, T, k = key
        ts = cells[key]
        n = len(ts)
        dchars = sorted({t["delivered_chars"] for t in ts})
        d_s = "/".join(str(x) for x in dchars) if len(dchars) <= 2 else f"{dchars[0]}--{dchars[-1]}"
        if strat == "naive_retry":
            con = frac(sum(bool(t["contradicts"]) for t in ts), n)
        else:
            con = "n/a"
        rows.append(f"{CONFIG_SHORT.get(label, tex(label))} & {STRATEGY_NAME.get(strat, tex(strat))} & "
                    f"{T:g} & {k} & {d_s} & {con} & "
                    f"{frac(sum(t['F_eq_G'] for t in ts), n)} & "
                    f"{frac(sum(t['F_eq_G2'] for t in ts), n)} \\\\")
    with open(os.path.join(a.out, "anomalies.tex"), "w", encoding="utf-8") as f:
        header(f)
        f.write("\n".join(rows) + "\n")

    # latency
    with open(os.path.join(a.out, "latency.tex"), "w", encoding="utf-8") as f:
        header(f)
        for (label, strat) in sorted(lat, key=lambda k: (order_key(k[0]), k[1])):
            ts = lat[(label, strat)]
            f.write(f"{CONFIG_SHORT.get(label, tex(label))} & {STRATEGY_NAME.get(strat, tex(strat))} & "
                    f"{len(ts)} & {med_range([t['restart_s'] for t in ts])} & "
                    f"{med_range([t['ttft_after_restart_s'] for t in ts], '{:.2f}')} & "
                    f"{med_range([t['recovery_latency_s'] for t in ts])} \\\\\n")

    # extra prompts: one row per (configuration, prompt)
    with open(os.path.join(a.out, "prompts.tex"), "w", encoding="utf-8") as f:
        header(f)
        for (label, pid) in sorted(prompt_repro, key=lambda k: (order_key(k[0]), k[1])):
            rs = prompt_repro[(label, pid)]
            nv = prompt_cells.get((label, pid, "naive_retry"), [])
            rc = prompt_cells.get((label, pid, "recompute"), [])
            f.write(f"{CONFIG_SHORT.get(label, tex(label))} & {tex(pid)} & {frac(sum(rs), len(rs))} & "
                    f"{frac(sum(bool(t['contradicts']) for t in nv), len(nv))} & "
                    f"{frac(sum(t['F_eq_G'] or t['F_eq_G2'] for t in rc), len(rc))} \\\\\n")

    # token-id recompute: does re-tokenizing the delivered text change the tokens?
    with open(os.path.join(a.out, "retok.tex"), "w", encoding="utf-8") as f:
        header(f)
        for key in sorted(cells, key=lambda k: (order_key(k[0]), k[1], k[2], k[3])):
            if key[1] != "recompute_ids":
                continue
            ts = cells[key]
            f.write(f"{CONFIG_SHORT.get(key[0], tex(key[0]))} & {key[2]:g} & {len(ts)} & "
                    f"{frac(sum(bool(t['retok_matches']) for t in ts), len(ts))} & "
                    f"{frac(sum(t['F_eq_G'] for t in ts), len(ts))} & "
                    f"{frac(sum(t['F_eq_G2'] for t in ts), len(ts))} \\\\\n")

    # environment
    with open(os.path.join(a.out, "env.tex"), "w", encoding="utf-8") as f:
        header(f)
        for label in sorted(env_by_label, key=order_key):
            e, _cmd = env_by_label[label]
            pk = e.get("packages", {})
            gpus = e.get("gpus") or []
            gpu = (f"{gpus[0]['name']} ({gpus[0]['memory_total']}), driver {gpus[0]['driver']}"
                   if gpus else "none recorded")
            engine_pkg = label.split("-")[0]
            f.write(f"{CONFIG_SHORT.get(label, tex(label))} & {tex(pk.get(engine_pkg) or 'n/a')} & "
                    f"{tex(pk.get('torch') or 'n/a')} & {tex(gpu)} \\\\\n")

    # provenance summary
    in_flight = [t["in_flight"] for _, ts in runs for t in ts if t["in_flight"] is not None]
    with open(os.path.join(a.out, "provenance.tex"), "w", encoding="utf-8") as f:
        header(f)
        f.write(f"{len(runs)} measured runs, {n_trials} trials, harness commit(s) "
                f"\\texttt{{{tex(', '.join(sorted(commits)))}}}. "
                f"In-flight chunks received after the kill: "
                f"{max(in_flight) if in_flight else '--'} at most, over all trials.\n")

    # Phase 3 gateway tables (analysis/gw_tables.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gw_tables
    gm = gw_tables.write(a.root, a.out)
    if gm:
        print("gateway: " + ", ".join(f"{k}={v}" for k, v in gm.items()))

    # Paper version 3: Phases 2, 4 and 5 (analysis/v3_tables.py)
    import v3_tables
    vm = v3_tables.write(a.root, a.out)
    print("v3: " + ", ".join(f"{k}={v}" for k, v in vm.items()))

    print(f"wrote {len(rows)} anomaly rows, {len(repro)} reproducibility rows, "
          f"{len(lat)} latency rows to {a.out}/")


if __name__ == "__main__":
    main()
