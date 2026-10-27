# Phase 5 (RQ4) plan, revised after Phase 4

Written before any Phase 5 result was computed. The decision rule at the end
decides whether the expensive live version (a real KV checkpoint inside vLLM,
"Option B") is worth building.

## Why the plan changed

The proposal's RQ4 assumed RQ3 would find a crossover: recompute for short
requests, checkpoint for long ones, and a policy choosing per request. Phase 4
([checkpoint.md](checkpoint.md)) found instead that restoring a KV copy from the
standby machine beats recomputing it at every measured length (1.9-3.1x with 4
TCP streams). So *whether* to checkpoint is nearly settled on this hardware for
a single request. What is not settled is **how often** to stream new KV entries
to the standby, because every checkpoint may slow decoding, and that cost was
not measured (raw copies ran on an idle GPU).

## Question and hypothesis

**RQ4 (restated).** Does a per-request adaptive checkpoint policy beat the best
fixed policy on a mixed workload, once checkpoint cost on the decode path is
accounted for?

**Hypothesis (from the proposal, unchanged).** An adaptive policy matches or
beats the best fixed interval across mixed workloads, because no single interval
suits both short chat turns and long-context requests.

## Option A: a simulator driven by measured costs

Clearly labeled as simulation, in its own directory (`simulation/`), never mixed
with measured results (proposal, provenance rules). Every hardware cost is read
from committed raw logs by [`policies/params.py`](../policies/params.py); nothing
measured is typed in by hand:

| Input | Source (committed logs) |
|---|---|
| Recompute cost R(n), 512-32,768 tokens | Phase 4 prefill runs (`*_prefill_*`) |
| KV bytes per token | `kv_bytes_per_token` in the Phase 4 copy runs |
| GPU<->host bandwidth | Phase 4 one-machine copy run (`*_kvcopy`) |
| Network bandwidth to the standby | Phase 4 transfer runs (`*_kvnet_s4`, `*_kvnet_s1`) |
| Time per output token | Phase 2 throughput runs (vLLM default, one request at a time) |
| Failover detection and switch time D | Phase 3 failover runs, minus the prefill they included |

**Assumptions** (not measured; swept or stated, and reported with every result):

* **Decode stall fraction beta**: the share of a checkpoint's copy time that
  stalls decoding. 0 = fully overlapped, 1 = fully blocking. Swept over
  0, 0.1, 0.5, 1. This is the quantity Option B would measure.
* **Failure rate**: mean time between failures of a serving replica of 1 day,
  1 hour, 10 minutes. Failures are fail-stop and Poisson; the expected cost uses
  a first-order approximation (at most one failure per request), accurate when
  failure rate x request duration is small.
* **Workloads**: *chat*, prompt 64-2,048 tokens (log-uniform), output 64-512;
  *long*, prompt 4,096-28,000 tokens (log-uniform), output 256-2,048, total
  context at most 32,768; *mixed*, 80% chat and 20% long. 2,000 requests each,
  fixed seed.
* One request at a time; time per output token constant over context length
  (Phase 2 measured it at short context only); the recompute of the tokens after
  the last checkpoint costs R(n) - R(n_ckpt); the standby holds the checkpoint in
  host memory and restores it at the measured host->GPU rate; a standby is always
  available.

**Policies.** Decisions are made at the end of prefill and every 16 output tokens
(vLLM's block size).

| Policy | Rule |
|---|---|
| recompute | never checkpoint (the Phase 3 gateway as built) |
| prompt-only | checkpoint the prompt's KV once, after prefill |
| fixed-k | checkpoint after prefill and every k output tokens, k = 16, 64, 256, 1024 |
| adaptive (proposal rule) | checkpoint if lambda * dt * R(n) > beta * copy time of the new entries, dt = time to the next decision (the proposal's inequality as written) |
| adaptive (remaining-time rule) | checkpoint if lambda * T_remaining * [saving per failure] > beta * copy time, where saving = R(n) - R(n_ckpt) minus the extra restore time, and T_remaining uses the request's max_tokens |

**Metrics** per policy, workload, beta and failure rate: expected completion
time per request (failure-free time plus expected recovery cost), its overhead
over recompute's failure-free time, and recovery latency (failure to next token)
over failure positions spread uniformly in time across each request: mean and
99th percentile.

## Decision rule for Option B

For each workload, beta and failure rate, the **baseline** is the best
non-adaptive policy (recompute, prompt-only or any fixed k) by expected
completion time, chosen in hindsight per configuration, which is stronger than
any single interval chosen in advance.

An adaptive policy **wins** a configuration if it either lowers expected
completion time by at least 5% against the baseline, or lowers 99th-percentile
recovery latency by at least 5% against the baseline while its expected
completion time is no more than 0.1% worse.

**Go to Option B** if an adaptive policy wins in the *mixed* workload for at
least one beta in {0.1, 0.5} at a failure rate of 1 hour or rarer. Otherwise
report the negative result, and the live work narrows to measuring beta for the
best fixed policy only.
