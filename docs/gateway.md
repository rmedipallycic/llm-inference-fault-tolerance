# Phase 3: the exactly-once token gateway (RQ2)

Status: **measured on vLLM 0.31.0** (one A10G, Llama 3.1 8B Instruct,
restart mode, Oct 21, 2026); SGLang and two-replica failover not yet measured.
Every number below is in [`experiments/gateway_summary_vllm.md`](../experiments/gateway_summary_vllm.md),
which `python analysis/gateway_check.py` regenerates from the raw logs.

## Why a gateway

Phase 1 measured what a streaming client sees when an engine dies mid-stream.
Naive retry contradicted text the client had already shown in most trials.
Recompute (resubmit the prompt plus the delivered tokens) never contradicted
the client, and together with the engines' deterministic modes it reproduced
the failure-free output exactly. Recompute, however, was done by the
*client*, and the client also had to drop duplicated positions itself.

The gateway moves that work out of the client. An unmodified client talks to
the gateway with the usual OpenAI-compatible streaming API and gets
exactly-once delivery of output positions, with no recovery code of its own.

## Protocol

```
client ──HTTP/SSE──► gateway ──HTTP/SSE──► engine replica 0 (vLLM / SGLang)
   ▲   seq, n            │   prompt ids +      engine replica 1 (optional)
   │                     │   delivered ids
   │                     ▼
   └── resume ◄── token log (one JSON-lines file per request)
```

1. **Numbering.** Each entry streamed to the client carries `seq`, the
   position of its first token in the output, and `n`, the number of tokens it
   holds (`gateway` field of each SSE event; the OpenAI fields are unchanged).
2. **Output commit.** The gateway appends each entry to the request's log
   before releasing it to the client (optionally with `fsync`). Everything a
   client has seen is therefore in the log.
3. **Engine failure.** A broken stream, an HTTP error or no chunk for
   `--stall-timeout` seconds counts as a failed attempt. The gateway picks a
   healthy replica, trying the others before the one that failed (so with two
   replicas it fails over; with one it waits for the restart, up to
   `--recovery-timeout`), and resubmits *prompt token ids + every committed
   token id* with `max_tokens` set to what is left. Nothing is re-tokenized.
4. **Client reconnect.** A client whose connection dropped calls
   `GET /v1/streams/<request id>?from_seq=<next position>` and receives exactly
   the entries it has not seen, then the rest of the generation.
5. **Gateway restart.** On start, the gateway resumes every logged request
   without a `done` entry, from its committed tokens; clients reconnect as in 4.

## What is guaranteed

| Property | Holds | Why |
| --- | --- | --- |
| No duplicated output positions at the client | always | positions are numbered; resumes ask only for the rest; reconnects start at `from_seq` |
| No missing positions | unless the request fails for good (e.g. no replica comes back) | every committed entry is in the log and is replayed on reconnect |
| Continuation never contradicts delivered text | always | the delivered token ids are the context for the continuation |
| Final text equals a failure-free run | only if the engine is deterministic | Phase 1: vLLM batch invariance, SGLang deterministic inference |
| Same *sample* at temperature > 0 | no | the engines' HTTP APIs do not expose the sampler's RNG position, so the proposal's "preserved sampler state" cannot be implemented without engine changes; the resumed part is a fresh sample |

Failure model: fail-stop failures of engine processes and of the gateway
process. The log directory must survive (local disk; a lost disk is out of
scope), and so must the client.

## Resume modes

* `ids` (default): needs the engine's `/tokenize` and token ids in the stream
  (vLLM: `logprobs: 1, return_tokens_as_token_ids: true`). `seq` counts tokens.
* `text`: prompt text + delivered text, for engines that do not return token
  ids. `seq` counts chunks. Phase 1 found that re-tokenizing delivered text
  gave the same ids in 10 of 10 vLLM trials, but this was not checked for
  SGLang.

## Tests (no GPU)

`python -m unittest discover -s tests -v` runs the mock engine
(`harness/mock_engine.py`, not a model) and the gateway as separate processes
and SIGKILLs them mid-stream:

| Test | Checks |
| --- | --- |
| failure-free | gateway output equals the engine's, no recovery attempts |
| engine killed and restarted | exactly-once; equals the failure-free output (deterministic mock) |
| failover to second replica | exactly-once; equals failure-free; recovers in under 2 s without a restart |
| history-dependent engine | exactly-once; delivered prefix kept although a cold engine generates differently |
| temperature 0.8 | exactly-once |
| text mode | exactly-once; equals failure-free |
| two failures in one request | exactly-once; two recovery attempts |
| client reconnect | the dropped client resumes with `from_seq`; exactly-once |
| gateway killed and restarted | unfinished request resumed from the log; exactly-once |
| negative control | without the gateway, naive retry repeats positions and the checker flags it |

## Measured results (vLLM 0.31.0, restart mode)

Grid: `bash scripts/run_gateway_matrix.sh vllm` at commit `8f7d849`, 256-token
budget, the Phase 1 prompt, seed 1234, 10 trials per run.

**Delivery.** In all 70 kill trials (T = 0.0 and 0.8, kill at 25, 128 and 230
tokens, plus 10 trials in batch-invariant mode) the client received every
output position exactly once: 0 duplicated and 0 missing positions, every
request completed, one recovery attempt each. Phase 1 measured naive retry,
with no gateway, contradicting the client in 25 of 30 vLLM trials.

**Same text as a failure-free run.** With `VLLM_BATCH_INVARIANT=1`, the
recovered output equalled the failure-free output in 10 of 10 trials (T = 0,
kill at 128), and the two failure-free references were identical. In the
default mode at T = 0, 2 of 30 recovered outputs equalled either failure-free
reference, which themselves differed (the first request after a start differs
from the second, as in Phase 1). At T = 0.8 the resumed part is a fresh
sample; recovered outputs equalled a reference in 0, 1 and 10 of 10 trials
for kills at 25, 128 and 230 tokens. With the seed fixed and few tokens left,
the restarted sampler apparently produced the same tail; this is not a
guarantee and is not reported as exact recovery.

**Recovery time is engine restart.** Kill to first recovered token was
107.6-108.4 s (median per run) in the first four runs and 173.0-184.8 s in the
last three. Restart (kill to the engine answering its health check) accounts
for all of it: the engine-ready-to-first-token gap was 0.17-0.49 s (median per
run). That gap is only known to about 1 s, because the harness and the gateway
poll the engine's health independently (every 1 s and 0.5 s), so a few
individual gaps come out slightly negative. The jump between runs comes from
vLLM's weight loading (`Loading weights took` in each `server_0.log`): 44.8-46.9 s
in all 44 engine starts before 02:33 UTC, and 68.9-125.8 s in 33 of the 35
starts after it (the other two took 46.0 and 46.3 s). In a fast and a slow start
compared line by line, torch.compile (0.4 s) and CUDA-graph capture took the same time. A likely cause is the
15 GB of weights dropping out of the 16 GiB machine's page cache, after which
they were read from the gp3 volume at its default 125 MB/s; this was not
verified. A second replica (failover) would remove the restart from recovery.

**Overhead** (30 failure-free pairs each, direct vs gateway, alternating
order; median of per-pair differences):

| | without fsync | with fsync |
| --- | --- | --- |
| time to first token | +38.8 ms (40.8 -> 79.6 ms) | +44.8 ms (40.9 -> 85.6 ms) |
| time per output token | +0.05 ms (33.11 -> 33.16 ms) | +0.06 ms (33.11 -> 33.17 ms) |
| end-to-end, 256 tokens | +0.05 s (8.48 -> 8.54 s) | +0.06 s (8.48 -> 8.54 s) |

The per-token cost is small, as the proposal's hypothesis expected. Most of
the cost is paid once per request, before the first token: this
implementation makes three extra round trips to the engine (two health checks
and a `/tokenize` call) and asks for token ids in logprobs. That is a cost of
this implementation, which could be cut (cached health state, client-side
tokenization), not a measured lower bound for exactly-once delivery.

## Measurements still to run (`scripts/run_gateway_matrix.sh`)

* **restart**: one replica, T = 0.0 and 0.8, kill at 10/50/90% of 256
  tokens, 10 trials each: delivery check, recovery latency, equality with the
  failure-free reference.
* **determ**: the same with the engine's deterministic mode at T = 0, where
  the expected result is the failure-free output in every trial.
* **overhead**: 30 failure-free pairs, direct vs gateway, without and with
  `fsync`: time to first token, time per output token, end-to-end time.
  Direct requests use a text prompt without logprobs, as a client without the
  gateway would; the gateway's own token-id requests are part of its cost.
* **failover**: two replicas on two GPUs (skipped on a single A10G, where two
  8B replicas do not fit). Not yet run.
* **SGLang**: the same grid in text mode. Not yet run.

The hypothesis from the proposal, that exactly-once delivery costs little per
token because the log grows by one entry per step, is what the overhead runs
test. It can be refuted.
