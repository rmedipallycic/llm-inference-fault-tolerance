# Fault-Tolerant LLM Inference Systems

**Recovery semantics, exactly-once token delivery, and adaptive KV-cache checkpointing for LLM serving engines (vLLM, SGLang).**

> **Status (October 12, 2026):** Phase 1 measured on vLLM 0.31.0 (AWS g5.xlarge, one A10G, Llama 3.1 8B Instruct): 60-trial grid plus a 20-trial prefix-cache ablation, raw logs in [`experiments/`](experiments/). Results are in the [paper draft](paper/main.pdf) and summarized below. SGLang could not be started on the machine image used and is deferred; see [the SGLang note](docs/sglang-attempt.md).

---

## Research Overview

LLM inference has become stateful streaming infrastructure. A single request holds a growing KV cache in GPU memory and streams tokens to a client for seconds to minutes. Yet when a serving worker fails mid-generation, the common recovery is stateless: resubmit the prompt and generate again.

This project studies LLM serving through the lens of fault tolerance in distributed stream processing. Systems like Flink and Spark Structured Streaming have spent years solving the same three problems: how often to checkpoint state, how to recover it, and how to give downstream consumers exactly-once output. The goal is to measure how much of that toolkit transfers to LLM inference, and where it breaks.

## Problem Statement

A worker failure during generation has two costs that are rarely measured together.

**Performance cost.** Recovery must rebuild the KV cache with a full prefill over the prompt plus every token already generated. Recovery time therefore grows with both prompt length and output length, which matters most for long-context and agentic workloads.

**Correctness cost at the client.** A client that has already received hundreds of tokens may, after a retry, receive them again, miss some, or receive an entirely different continuation. With sampling temperature above 0, a restart is not reproducible unless sampler state is preserved. Most serving stacks give no explicit delivery guarantee across failures.

## Research Questions

| ID | Question |
| --- | --- |
| RQ1 | What does a streaming client observe when a vLLM or SGLang worker fails mid-generation under default recovery? |
| RQ2 | What latency and throughput does exactly-once token delivery cost? |
| RQ3 | At what context length, if any, does checkpointing KV state beat recomputing it? |
| RQ4 | Does an adaptive, per-request checkpoint policy beat fixed-interval policies on mixed workloads? |

Each question has a stated hypothesis in [`docs/proposal.md`](docs/proposal.md) written so that a measured result could refute it. If RQ3 finds no crossover at realistic context lengths, that negative result will be reported as such.

## System Architecture Under Study

```
                 ┌──────────────────────────────────────────────┐
  Client  ◄──────┤  Gateway (planned)                           │
 (streaming)     │  - sequence-numbered tokens                  │
                 │  - durable token log (output commit)         │
                 │  - dedup on recovery                         │
                 └───────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────▼──────────────────────────────┐
                 │  Serving engine: vLLM or SGLang              │
                 │  - continuous batching                       │
                 │  - block-managed KV cache on GPU             │
                 │  - checkpoint hook (planned) ──► host / peer │
                 └───────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────▼──────────────────────────────┐
                 │  Fault injector (built, harness/)            │
                 │  - kills worker at a chosen token position   │
                 │  - records full client-side token stream     │
                 └──────────────────────────────────────────────┘
```

**Recovery strategies compared**

| Strategy | Role | What happens on failure |
| --- | --- | --- |
| Naive retry | Baseline | Resubmit the original prompt only |
| Recompute | Baseline | Resubmit prompt + delivered tokens, rebuild KV cache by prefill |
| Fixed-interval KV checkpoint | Baseline | Restore last checkpoint (every *k* tokens), recompute the tail |
| Adaptive KV checkpoint | Proposed | Checkpoint only when expected recompute saved exceeds checkpoint cost |

## Failure Taxonomy

| Failure | Example | Detection | In scope? |
| --- | --- | --- | --- |
| Process crash | Worker OOM, segfault, SIGKILL | Connection drop, health check | Yes (primary) |
| GPU failure | Xid error, GPU reset | Driver error, health check | Yes (emulated by process kill) |
| Node loss | Instance termination | Heartbeat timeout | Planned, later phase |
| Preemption with warning | Spot reclaim notice | Cloud notice | Compared against prior work only |
| Hang / straggler | Stuck kernel, NCCL hang | Timeout | Out of scope for now |
| Silent data corruption | Bit flip in KV cache | Not detectable by fail-stop methods | Out of scope |

## Experimental Framework

**Planned setup**

| Component | Choice |
| --- | --- |
| Serving engines | vLLM and SGLang |
| Model | Llama 3.1 8B Instruct (proposed) |
| Hardware | Single GPU with 24 GB or more to start; multi-GPU tensor parallel in a later phase |
| Workloads | Short chat turns, long-context requests, and a mixed trace; temperature 0 and above 0 run separately |
| Fault injection | SIGKILL at fixed token positions (10%, 50%, 90% of target output length) |

**Metrics**

| Metric | Unit | Answers |
| --- | --- | --- |
| Recovery latency (kill to next delivered token) | ms | RQ3, RQ4 |
| Wasted compute (tokens decoded more than once) | tokens | RQ3, RQ4 |
| Checkpoint overhead on failure-free requests | % change in time per output token | RQ2, RQ3 |
| Client anomalies (duplicate, missing, divergent tokens) | count per failed request | RQ1, RQ2 |
| Goodput under injected failure rate | requests/s meeting latency target | RQ4 |

**Provenance rules (apply to everything in this repo)**

1. Every reported number links to a raw log in `experiments/` plus the command and commit hash that produced it.
2. No hardcoded outcome parameters. Any simulation lives in a separate, labeled directory and is never mixed with measured results.
3. All statistics come from one script that reads raw logs and that anyone can re-run.
4. Planned work is never described in past tense.

## Key Findings (preliminary)

Measured on vLLM 0.31.0, one A10G, Llama 3.1 8B Instruct, one prompt, 256-token budget, requests issued one at a time. Every number comes from [`analysis/make_tables.py`](analysis/make_tables.py) run on the logs in [`experiments/`](experiments/) (harness commit `90da903`). Scope limits are in the paper's threats-to-validity section.

1. **Failure-free output was not unique under default settings.** The first request after a server start and the second identical request differed in all 12 runs: at character 506 of about 1,280 at temperature 0, and at character 9 at temperature 0.8 with a fixed seed. Both texts were exactly repeatable, so output depended on what the server had served before, not on the prompt alone.
2. **Naive retry contradicted the client.** A restarted server is cold, so the retry always returned the first-request text (30 of 30 trials) and disagreed with text the client had already shown in 25 of 30. Only the earliest kill at temperature 0 gave pure, removable duplication.
3. **Recompute never contradicted the client but changed the answer.** Its final text matched neither failure-free output in 24 of 30 trials.
4. **Automatic prefix caching was necessary for the history dependence.** With `--no-enable-prefix-caching`, the two failure-free outputs matched in 4 of 4 runs and naive retry never contradicted the client (0 of 10). Recompute at temperature 0 still departed from the reference 43 characters after the kill in 5 of 5 trials.
5. **Recovery latency was server restart.** Restart took 107 to 195 s; the first recovered token arrived 0.23 to 0.32 s after the server was ready.

| Phase | Gate to pass before the next phase | Status |
| --- | --- | --- |
| 1. Harness and baselines | Fault injection runs end to end on unmodified vLLM and SGLang, logs committed | Done for vLLM (16 runs, 80 trials committed); SGLang deferred ([note](docs/sglang-attempt.md)) |
| 2. RQ1 characterization | Measured anomaly counts for naive retry and recompute | Done for vLLM, one prompt ([paper](paper/main.pdf)); more prompts and SGLang pending |
| 3. Exactly-once gateway (RQ2) | Zero client anomalies across injected trials, overhead measured | Not started |
| 4. Checkpoint microbenchmarks (RQ3) | Measured recompute-cost and checkpoint-cost curves | Not started |
| 5. Adaptive policy (RQ4) | Compared against all baselines on mixed workload | Not started |

## Repository Structure

```
llm-inference-fault-tolerance/
├── README.md
├── LICENSE                  # Apache-2.0
├── .gitignore
├── docs/
│   ├── proposal.md          # full research proposal
│   ├── harness-validation.md # smoke-test note
│   └── sglang-attempt.md    # why SGLang is deferred
├── harness/                 # fault-injection harness (Phase 1)
├── gateway/                 # exactly-once token gateway (planned)
├── policies/                # fixed and adaptive checkpoint policies (planned)
├── experiments/             # raw logs from measured runs only (16 runs, 80 trials)
├── analysis/                # analyze.py (per-trial metrics) and make_tables.py (paper tables), both from raw logs
├── scripts/run_matrix.sh    # runs the full measurement grid for one engine
├── paper/                   # paper draft with measured vLLM results
└── smoke_tests/summary.csv  # harness smoke test only, not results
```

## How to Reproduce

Setup for a laptop smoke test and for GPU runs (AWS or any Linux GPU machine) is in [`harness/README.md`](harness/README.md). To regenerate the paper's tables from the committed logs (no GPU needed):

```bash
python analysis/make_tables.py           # every paper table, from experiments/
cd paper && latexmk -pdf main.tex        # paper picks up the generated tables
```

To re-run the measurements on a GPU machine with a Hugging Face token that has Llama 3.1 access:

```bash
bash scripts/setup_instance.sh vllm      # Python 3.12 env, vLLM, model download, 1-trial smoke pilot
TRIALS=5 bash scripts/run_matrix.sh vllm # main grid: 12 runs x 5 trials
```

The prefix-cache ablation used the same harness with `--engine vllm-noapc` and `--no-enable-prefix-caching` added to the server command, at `--kill-at 128` for both strategies and temperatures; the exact command line is recorded in each ablation run's `run.json`.

Recorded environment: vLLM 0.31.0, torch 2.13.0, NVIDIA A10G (23028 MiB), driver 595.91.07, Deep Learning OSS Nvidia Driver AMI (Ubuntu 26.04, build 20261003), Python 3.12, `VLLM_USE_FLASHINFER_SAMPLER=0` (the image has no CUDA toolkit). Full package versions and GPU inventory are in every `run.json`.

## Related Work

- **DéjàVu** (Strati et al., ICML 2024) streams and replicates KV cache for fault-tolerant LLM serving. This project does not claim KV replication as new. It focuses on client-visible delivery semantics across failures and per-request adaptive checkpoint decisions. [Paper](https://proceedings.mlr.press/v235/strati24a.html)
- **SpotServe** (ASPLOS 2024) and **Llumnix** (OSDI 2024) migrate in-flight request state across instances, for preemption and load balancing respectively. Both assume planned moves; this project targets unplanned crashes.
- **Defeating Nondeterminism in LLM Inference** (He et al., Thinking Machines Lab, 2025) shows that inference at temperature 0 varies with server load because kernels are not batch-invariant. The finding above is the fault-tolerance consequence: a retry after a crash is not a replay.
- **vLLM / PagedAttention** (SOSP 2023) and **SGLang** provide block-managed KV caches, which make block-granular checkpointing feasible.
- **Rollback-recovery and stream processing:** Chandy and Lamport (1985) on consistent snapshots, Elnozahy et al. (2002) on rollback-recovery and output commit, Young (1974) and Daly (2006) on optimal checkpoint intervals, and the checkpointing and exactly-once models of Apache Flink and Spark Structured Streaming.

## Citation

A paper draft with the measured vLLM results is in [`paper/`](paper/main.pdf). To reference this repository:

```bibtex
@misc{medipally2026llmft,
  author       = {Medipally, Rajshekar},
  title        = {What Does a Streaming Client See When an {LLM} Server Crashes? A Fault-Injection Study of Client-Visible Token Anomalies in {vLLM}},
  year         = {2026},
  howpublished = {\url{https://github.com/rmedipallycic/llm-inference-fault-tolerance}},
  note         = {Preprint draft}
}
```

## Author

**Rajshekar Medipally**, Data Engineer and Cloud Architect, Richmond, VA
Research interests: distributed systems, fault tolerance, ML systems infrastructure
rmedipallycic@gmail.com · [github.com/rmedipallycic](https://github.com/rmedipallycic)

## License

Apache License 2.0. See [LICENSE](LICENSE).
