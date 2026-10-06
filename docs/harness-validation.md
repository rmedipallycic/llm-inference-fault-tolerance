# Harness validation (smoke test, October 2026)

**This is a check that the Phase 1 harness works, not a measurement.** It used a
0.5B model on a laptop CPU with 2 trials per configuration. None of these numbers
are findings, and none will be reported as results. Measured runs will come from
vLLM and SGLang on a GPU and will live in `experiments/`.

The per-trial rows are in [`smoke_tests/summary.csv`](../smoke_tests/summary.csv).

## Setup

| Item | Value |
| --- | --- |
| Server | llama.cpp `llama-server`, build b11451, Windows CPU |
| Model | Qwen2.5-0.5B-Instruct, GGUF q4_k_m |
| Request | 128 max tokens, server killed after 40 chunks |
| Trials | 2 per configuration, 3 configurations |
| Harness commits | `f7abb17` (naive retry runs), `7e70b0d` (recompute run) |

## What was checked

- The harness starts the server, kills its whole process tree at the chosen chunk,
  restarts it, and runs the recovery request, on Windows.
- Raw logs record every chunk with timestamps, the git commit, and the server log.
- Both recovery strategies run end to end, and `analysis/analyze.py` reads the raw
  logs back.

## What it showed about the harness and the analysis

1. **At temperature 0, the two strategies behave as designed.** Naive retry
   re-sent all of the already-delivered text, and recompute continued from it, so
   the client's combined output matched the failure-free reference.
2. **At temperature 0.8, llama-server's output was not reproducible across
   requests, even with a fixed seed.** The first request after each server start
   matched, and later requests did not. This suggests the seed acts on server-wide
   state rather than per request (a hypothesis, not confirmed in llama.cpp's code).
   The analysis now warns when two clean reference runs disagree, because
   divergence cannot be interpreted in that case.
3. **The first duplication metric was too loose.** It counted a single shared
   character as duplication. It was replaced by `duplicated_fraction`, the share
   of delivered text that was sent again.

## Next

Repeat on vLLM with Llama 3.1 8B on a GPU. The first thing to check there is
whether two clean seeded runs agree, which decides whether temperature above 0
can be measured on that engine.
