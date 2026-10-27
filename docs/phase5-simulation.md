# Phase 5 (RQ4), Option A: simulation results

**SIMULATION, not measurement.** The model, its assumptions and the decision rule
are in [phase5-plan.md](phase5-plan.md), committed before the simulator was run.
Hardware costs are read from the committed Phase 2-4 logs by
[`policies/params.py`](../policies/params.py); everything else is an assumption
stated there. Full tables: [`simulation/summary.md`](../simulation/summary.md);
raw output: [`simulation/results.json`](../simulation/results.json); rerun with
`python -m policies.run_sim` (about 4 minutes, no GPU).

## Decision: no-go for an adaptive policy

By the rule written beforehand, neither adaptive policy won a single
configuration, in any workload. The live version (Option B) should therefore not
build an adaptive policy; per the plan, it narrows to measuring the decode-path
cost of the best fixed policy.

**Why, in one line:** at the failure rates simulated, the expected cost of
failures is too small for any checkpoint to pay for itself in average completion
time. At a mean time between failures of 1 hour, a 15.5 s mixed-workload request
fails with probability about 0.4%, so even removing all recovery time can save at
most a few milliseconds per request on average, while a checkpoint that stalls
decoding for 10% of its copy time already costs 0.2-0.4%. At failure rates of
1 hour or rarer, both adaptive rules work this out and stop checkpointing
entirely as soon as checkpoints cost anything (beta >= 0.1); they then *match*
the best non-adaptive policy, which is recompute, but cannot beat it. At the
10-minute rate with beta 0.1 the best non-adaptive policy is prompt-only (mixed)
or fixed-16 (long), the remaining-time rule checkpoints a little (6.3 times per
mixed request), and its expected completion time is within 0.1% of that
baseline; neither adaptive rule gains 5% anywhere.

So the proposal's RQ4 hypothesis, "matches or beats the best fixed interval", is
met only in the trivial sense of matching: adaptivity buys nothing measurable in
expected completion time. The decision rule required a 5% gain.

## What checkpointing does buy: recovery latency

The pre-registered metrics hide one effect the simulation shows clearly. For
failures **during decoding**, checkpointing cuts the time from failure to the
next token sharply (mixed workload, MTBF 1 hour, mean / p99 in seconds; added
after the run, outside the decision rule):

| Policy | Failure-free overhead, beta 0.1 / 0.5 / 1 | Recovery after a decode failure, mean / p99 |
|---|---|---|
| recompute | 0 | 0.98 / 9.08 |
| prompt-only | 0.21% / 1.07% / 2.14% | 0.26 / 1.04-1.07 |
| fixed-16 | 0.25% / 1.23% / 2.47% | 0.17-0.20 / 0.40-0.51 |
| fixed-1024 | 0.22% / 1.11% / 2.22% | 0.25 / 0.76-0.78 |

For long requests alone, recompute's p99 is 10.5 s. This is a tail-latency
argument, not an average-cost one: a deployment with a latency target for
recovery, not just for average completion, would checkpoint, and **prompt-only
captures most of the gain** (p99 from 9.1 s to about 1.05 s) for one copy per
request. A fixed interval adds the rest.

Two limits on that gain, both visible in the simulation:

* **Failures during prefill** (3.8% of failure positions in the mixed workload,
  11.1% in long) are recovered by redoing the prefill under every policy, so the
  pre-registered p99 over all failure positions barely moves (9.3 s for
  recompute, 6.1-7.0 s with checkpoints). Checkpointing chunks of a long prefill
  as they are computed would address this; no policy here does.
* **The prompt's checkpoint takes seconds to copy** for long prompts (about 3 s
  for 28,000 tokens at 9.9 Gbit/s), and a failure before it completes still
  needs a full recompute. That is why the long-workload p99 stays at 2.6-7.9 s
  even with checkpoints.

## What this does and does not show

* It does not show what a checkpoint costs on the decode path (beta). That is
  the one number that decides whether the 0.2% or the 2.5% overhead is real, and
  only a real engine integration can measure it.
* The model is first-order (at most one failure per request; the largest
  failure rate x duration in the grid is 0.13), one request at a time, constant
  time per token across context lengths, and a tail recompute cost of
  R(n) - R(n_ckpt). Concurrency would change both the stall and the copy rate.
* Failure rates of 1 hour and 10 minutes per replica are far higher than typical
  GPU servers see; at realistic rates the expected-cost argument against
  checkpointing is stronger still, and only the tail-latency argument remains.

## What Option B becomes

Not an adaptive policy. The live work worth doing is: stream the prompt's KV
(and optionally every k tokens) to the standby from inside vLLM, measure the
decode slowdown beta under load, and inject failures to measure recovery latency
against the Phase 3 gateway's recompute. If beta is near 0.1 or below, prompt-only
or fixed-interval checkpointing is a cheap way to cut recovery tail latency by
roughly an order of magnitude for long contexts; if it is near 1, the overhead
(about 2-2.5%) has to be weighed against that tail.
