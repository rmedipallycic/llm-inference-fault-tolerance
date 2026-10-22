# Phase 3: the exactly-once token gateway (RQ2)

Status: **measured** on vLLM 0.31.0 and SGLang 0.5.21 (AWS g5.xlarge, one
A10G each, Llama 3.1 8B Instruct, Oct 21-22, 2026): restart recovery on both
engines and failover between two machines on vLLM, 190 kill trials in all.
Every number below is in [`experiments/gateway_summary.md`](../experiments/gateway_summary.md),
which `python analysis/gateway_check.py` regenerates from the raw logs.

**Summary.**
* Exactly-once delivery held in all 190 kill trials: 0 duplicated and 0
  missing output positions, every request completed.
* With the engines' deterministic modes, restart recovery reproduced the
  failure-free text in 10 of 10 trials on each engine.
* Failing over to a replica on a second machine cut recovery from 108-185 s
  (engine restart) to 0.16-0.17 s (median per run).
* Exact text after failover needed more than the same software: with each
  machine's own compiled-kernel cache it matched in 0 of 20 batch-invariant
  trials; with the first machine's caches copied to the second, in 10 of 10.
* The per-token cost of the gateway was below measurement resolution on SGLang
  and +0.05 ms (0.15%) on vLLM; about 32-45 ms are added once per request.

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
| Same *sample* at temperature > 0 | vLLM 0.31 with a fixed request `seed`, if the arithmetic matches; not SGLang 0.5.21 in its default mode | vLLM 0.31's sampler draws its noise from a hash of the request seed, the token position and the token id (`vllm/v1/worker/gpu/sample/gumbel.py`), so a resumed request, whose tokens sit at the same positions, draws the same noise; it keeps no running RNG state. Without an explicit seed vLLM picks a random one per request, so the resumed part is a fresh sample. SGLang ignored the OpenAI `seed` field in Phase 1, and its default-mode samples differed in every trial here |

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
from the second, as in Phase 1). At T = 0.8, recovered outputs equalled a
reference in 0, 1 and 10 of 10 trials for kills at 25, 128 and 230 tokens.
With a fixed seed, vLLM 0.31 resamples the same noise after a resume (see the
guarantee table), so the remaining differences come from the arithmetic: the
later the kill, the fewer tokens are left in which a small numerical
difference after the restart can change a choice.

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

## Measured results: failover between two machines (vLLM 0.31.0)

Two g5.xlarge machines, same AMI, driver and vLLM 0.31.0 (setup below).
Each trial starts on replica 0 (machine 1), which is SIGKILLed at position 128;
the gateway resumes on replica 1 (machine 2). 10 trials per run, seed 1234.
Run conditions that the records cannot capture themselves are listed in
[`experiments/run_conditions.json`](../experiments/run_conditions.json), with
evidence in `experiments/pip_freeze_machine1.txt` and `experiments/failover_machine2/`.

| Run | Engine mode | Replica 1 setup | Exactly-once | Same text as failure-free | Recovery, median (range) |
| --- | --- | --- | --- | --- | --- |
| T = 0.0 | default | own cache, packages differ | 10/10 | 2/10 | 0.17 s (0.16-0.39) |
| T = 0.8 | default | own cache, packages differ | 10/10 | 10/10 | 0.16 s (0.16-0.17) |
| T = 0.0 | batch-invariant | own cache, packages differ | 10/10 | 0/10 | 0.17 s (0.16-0.41) |
| T = 0.0 | batch-invariant | own cache, packages identical | 10/10 | 0/10 | 0.17 s (0.16-0.40) |
| T = 0.0 | batch-invariant | machine 1's caches, packages identical | 10/10 | **10/10** | 0.17 s (0.16-0.41) |

Recovery is kill to the next position at the client; no engine restart is on
the path, so it is the time to notice the broken stream, check the other
replica's health and prefill the prompt plus the delivered tokens there.

**Why the text differed, and what fixed it.** In batch-invariant mode each
machine was repeatable on its own (its two failure-free references were
identical), but machine 2's failure-free answer differed from machine 1's
from character 554 on, in both own-cache runs: a single word ("undoing the
crash" against "undoing the effects of the crash") after which the two texts
continue differently, the pattern of a near-tie decided by a last-digit
difference in the arithmetic. It was not the batch-invariance setting (the
replica's process environment had `VLLM_BATCH_INVARIANT=1`) or the packages
(both machines: NVIDIA A10G, driver 595.91.07, same AMI; machine 2's VBIOS
version was not recorded, but the hardware is ruled out by the cache result
below):
before the second run, the 21 packages that differed (20 by version, mostly
web and HTTP libraries but also cuda-tile, cuda-bindings, cuda-core and
cuda-python, and z3-solver missing) were set to machine 1's versions, making
`pip freeze` byte-identical, and the answer still split at character 554.
What still differed was each machine's own compiled state: vLLM's
torch.compile cache (`~/.cache/vllm`, 235 MB on machine 1, 115 MB on machine 2),
FlashInfer's cache (`~/.cache/flashinfer`, 792 KB against 12 KB) and
TorchInductor's (`/tmp/torchinductor_ubuntu`). With machine 1's three caches
copied to machine 2, machine 2's failure-free answer was identical to machine
1's and all 10 failover trials reproduced the failure-free text. Which of the
three caches matters, and why compiled artifacts built on identical machines
differ, was not isolated. In default mode at T = 0.8 the failover outputs
matched a reference in 10 of 10 trials even with machine 2's own cache; a
sampled token is less sensitive to a last-digit difference than a greedy
choice at a near-tie, but 10 trials do not show how much less.

**Consequence for failover.** A standby replica gives exactly-once delivery
and sub-second recovery regardless; it gives the same text as a failure-free
run only if it runs in a deterministic mode *and* was started from the same
compiled state as the replica it replaces.

## Measured results (SGLang 0.5.21, restart mode)

Grid: `PARTS="restart determ overhead" bash scripts/run_gateway_matrix.sh sglang`
inside the Phase 1 container image (`lmsysorg/sglang:v0.5.21`, image id
`b1259f3ea327`), commit `7abc6ea`, text resume mode, otherwise as for vLLM.

* **Delivery:** exactly-once in 70 of 70 kill trials (0 duplicated, 0 missing
  positions, all requests complete, one recovery attempt each).
* **Same text:** with `--enable-deterministic-inference`, 10 of 10 (T = 0,
  kill at 128). Default mode: 3 of 30 at T = 0; 0 of 30 at T = 0.8, where all
  30 recovered texts differed from each other and the two references already
  differed at the second character, consistent with SGLang ignoring the OpenAI
  `seed` field (Phase 1).
* **Recovery:** 129.8-137.8 s (median per run) in default mode and 178.8 s in
  deterministic mode, again all SGLang restart: the engine-ready-to-first-token
  gap was -0.31 to 0.24 s (median per run; known only to about 1 s, as above).
* **Overhead** (30 pairs each): time to first token +31.7 ms (70.1 -> 101.8 ms)
  and +34.6 ms with fsync (64.7 -> 101.8 ms); time per output token
  32.81 -> 32.69 and 32.70 ms; end-to-end, 256 tokens, 8.44 -> 8.44 s and
  8.43 -> 8.44 s. So no measurable per-token or end-to-end cost; why the added
  time to first token does not show up end to end was not examined.

## Measurement plan (`scripts/run_gateway_matrix.sh`)

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
  8B replicas do not fit); run instead across two machines, below.
* **SGLang**: the same grid in text mode, from the container, below.

### Failover across two machines

Instances with two or more GPUs need a larger vCPU quota than this project
has, so failover runs on two single-GPU machines (two g5.xlarge, same AMI and
vLLM 0.31.0, same security group with TCP 8000 allowed from that group).
Machine 1 runs the gateway and replica 0, machine 2 runs replica 1:

```bash
# machine 2, in tmux (default mode first; later restart it with "bi")
bash scripts/remote_replica.sh default
# machine 1, in tmux, with machine 2's private IP
bash scripts/run_failover_two_hosts.sh default http://<private IP>:8000
```

Each trial starts on replica 0, which is SIGKILLed at position 128; the
gateway resumes on replica 1 over the private network. The run records
replica 1's `/version`, a failure-free reference generated on it, and the
command it was started with (`scripts/failover_common.sh` defines it for both
machines). Copy `~/remote_replica_<mode>.log` from machine 2 into the run's
folder before committing. To make the replicas compiled-state identical, copy
`~/.cache/vllm`, `~/.cache/flashinfer` and `/tmp/torchinductor_$USER` from
machine 1 to machine 2 before starting replica 1.

### SGLang

As in Phase 1, SGLang runs from its official container with the harness
inside it, so that the kill reaches the server:

```bash
docker run --gpus all --rm --network host --ipc=host --shm-size 16g \
  -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD":/work -w /work \
  lmsysorg/sglang:v0.5.21 bash -c \
  'git config --global --add safe.directory /work && PARTS="restart determ overhead" bash scripts/run_gateway_matrix.sh sglang'
```

The hypothesis from the proposal, that exactly-once delivery costs little per
token because the log grows by one entry per step, is what the overhead runs
test. It can be refuted.
