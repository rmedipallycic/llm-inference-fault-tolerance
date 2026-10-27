"""Phase 5 (RQ4) simulator: KV checkpoint policies for one request at a time.

SIMULATION, not measurement (docs/phase5-plan.md). Hardware costs come from
committed logs via policies/params.py; everything else is an assumption listed
in the plan and passed in explicitly.

Model of one request (prompt P tokens, max_tokens O):
  * prefill takes R(P); then O tokens are decoded at a constant time per token.
  * Checkpoint decisions are made after prefill and every BLOCK output tokens.
    A checkpoint copies the KV entries added since the last one (m tokens,
    m * kv_bytes) GPU -> host -> network to the standby's host memory; it takes
    copy(m) = m*kv_bytes*(1/d2h_bw + 1/net_bw), of which beta*copy(m) stalls
    decoding. It counts once the copy has finished.
  * A fail-stop failure at time tau loses the replica. The standby resumes after
    detect_s; it restores the last completed checkpoint (n_c tokens) at the
    host->GPU rate and recomputes the rest by prefill, R(n) - R(n_c); without a
    checkpoint it recomputes R(n). Delivered tokens are kept (the Phase 3
    gateway). A failure during prefill loses the prefill done so far.
  * Expected completion time = failure-free time T + lam * T * mean added time
    over failure positions spread uniformly in [0, T] (first order: at most one
    failure per request).
"""
import math
import random

BLOCK = 16


def copy_s(p, m, streams):
    return m * p["kv_bytes"] * (1.0 / p["d2h_bw"] + 1.0 / p["net_bw"][streams])


def restore_s(p, n_c):
    return n_c * p["kv_bytes"] / p["h2d_bw"]


# ---------------- policies ----------------
# decide(ctx) -> True to checkpoint now. ctx: n (context tokens), n_c (last
# checkpointed position, 0 if none), g (output tokens so far), P, O, lam, beta,
# p (params), R (recompute function), streams.

def never(ctx):
    return False


def prompt_only(ctx):
    return ctx["g"] == 0


def fixed(k):
    def rule(ctx):
        return ctx["g"] == 0 or ctx["g"] % k == 0
    rule.__name__ = f"fixed-{k}"
    return rule


def adaptive_proposal(ctx):
    """The proposal's inequality: lam * dt * R(n) > C(n - n_ckpt), with C the
    decode stall the checkpoint causes and dt the time to the next decision."""
    p = ctx["p"]
    stall = ctx["beta"] * copy_s(p, ctx["n"] - ctx["n_c"], ctx["streams"])
    dt = BLOCK * p["tpot"]
    return ctx["lam"] * dt * ctx["R"](ctx["n"]) > stall


def adaptive_remaining(ctx):
    """Checkpoint if the expected saving over the rest of the request exceeds the
    stall: lam * T_remaining * saving > stall, saving = recompute avoided minus
    the extra restore time."""
    p, R = ctx["p"], ctx["R"]
    n, n_c = ctx["n"], ctx["n_c"]
    stall = ctx["beta"] * copy_s(p, n - n_c, ctx["streams"])
    saving = (R(n) - R(n_c)) - (restore_s(p, n) - restore_s(p, n_c))
    t_rem = (ctx["O"] - ctx["g"]) * p["tpot"]
    return ctx["lam"] * t_rem * saving > stall


POLICIES = {"recompute": never, "prompt-only": prompt_only,
            **{f"fixed-{k}": fixed(k) for k in (16, 64, 256, 1024)},
            "adaptive-proposal": adaptive_proposal,
            "adaptive-remaining": adaptive_remaining}


# ---------------- one request ----------------

def timeline(p, R, P, O, rule, lam, beta, streams):
    """Failure-free run. Returns T, the decode segments, and checkpoint events
    (completion time, tokens covered)."""
    t = R(P)
    prefill_end = t
    n_c_decided = 0
    ckpts = []                 # (t_complete, n_covered)
    segs = []                  # (t0, t1, n0): decode of tokens n0.. at tpot each
    stall_total = 0.0
    g = 0
    while True:
        n = P + g
        if n > n_c_decided:
            ctx = {"n": n, "n_c": n_c_decided, "g": g, "P": P, "O": O, "lam": lam,
                   "beta": beta, "p": p, "R": R, "streams": streams}
            if rule(ctx):
                c = copy_s(p, n - n_c_decided, streams)
                ckpts.append((t + c, n))
                stall = beta * c
                t += stall
                stall_total += stall
                n_c_decided = n
        if g >= O:
            break
        step = min(BLOCK, O - g)
        segs.append((t, t + step * p["tpot"], n))
        t += step * p["tpot"]
        g += step
    return {"T": t, "prefill_end": prefill_end, "segs": segs, "ckpts": ckpts,
            "stall": stall_total, "n_ckpt": len(ckpts)}


def added_time(p, R, P, tl, tau):
    """Time a failure at tau adds: detection, restore, recompute (and any prefill
    already done, if during prefill). Equals the recovery latency to the next
    token, except for the lost prefill."""
    d = p["detect_s"]
    if tau < tl["prefill_end"]:
        return d + R(P) + tau, d + R(P)
    n = P
    for t0, t1, n0 in tl["segs"]:
        if tau < t0:
            n = n0
            break
        if tau < t1:
            n = n0 + int((tau - t0) / p["tpot"])
            break
        n = n0 + round((t1 - t0) / p["tpot"])
    n_c = 0
    for tc, nn in tl["ckpts"]:
        if tc <= tau:
            n_c = nn
    rec = d + (restore_s(p, n_c) + R(n) - R(n_c) if n_c else R(n))
    return rec, rec


def evaluate(p, R, reqs, rule, lam, beta, streams, positions=64):
    T, ff, rec_lat, rec_dec, exp_T, stalls, nck = [], [], [], [], [], [], []
    for P, O in reqs:
        tl = timeline(p, R, P, O, rule, lam, beta, streams)
        adds = []
        for i in range(positions):
            tau = (i + 0.5) / positions * tl["T"]
            a, r = added_time(p, R, P, tl, tau)
            adds.append(a)
            rec_lat.append(r)
            if tau >= tl["prefill_end"]:
                rec_dec.append(r)
        exp_T.append(tl["T"] + lam * tl["T"] * sum(adds) / len(adds))
        T.append(tl["T"])
        stalls.append(tl["stall"])
        nck.append(tl["n_ckpt"])
    rec_lat.sort()
    rec_dec.sort()
    return {"mean_expected_T": sum(exp_T) / len(exp_T),
            "mean_failure_free_T": sum(T) / len(T),
            "mean_stall": sum(stalls) / len(stalls),
            "mean_checkpoints": sum(nck) / len(nck),
            "recovery_mean": sum(rec_lat) / len(rec_lat),
            "recovery_p99": rec_lat[min(len(rec_lat) - 1, int(0.99 * len(rec_lat)))],
            "decode_recovery_mean": sum(rec_dec) / len(rec_dec),
            "decode_recovery_p99": rec_dec[min(len(rec_dec) - 1, int(0.99 * len(rec_dec)))],
            "share_in_prefill": 1 - len(rec_dec) / len(rec_lat),
            "max_lam_T": lam * max(T)}


# ---------------- workloads ----------------

def workload(name, n, seed, max_ctx):
    rng = random.Random(f"{name}-{seed}")

    def loguni(a, b):
        return int(math.exp(rng.uniform(math.log(a), math.log(b))))

    def chat():
        return loguni(64, 2048), rng.randint(64, 512)

    def long_():
        while True:
            P, O = loguni(4096, 28000), rng.randint(256, 2048)
            if P + O <= max_ctx:
                return P, O

    if name == "chat":
        return [chat() for _ in range(n)]
    if name == "long":
        return [long_() for _ in range(n)]
    if name == "mixed":
        return [chat() if rng.random() < 0.8 else long_() for _ in range(n)]
    raise ValueError(name)
