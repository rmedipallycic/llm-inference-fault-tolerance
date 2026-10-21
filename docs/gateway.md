# Phase 3: the exactly-once token gateway (RQ2)

Status: **built and tested against a mock engine; no GPU measurements yet.**
Nothing in this file is a measured result on vLLM or SGLang.

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

## Measurements planned on GPU (`scripts/run_gateway_matrix.sh`)

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
  8B replicas do not fit).

The hypothesis from the proposal, that exactly-once delivery costs little per
token because the log grows by one entry per step, is what the overhead runs
test. It can be refuted.
