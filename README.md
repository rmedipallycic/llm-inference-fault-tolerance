# Fault-Tolerant LLM Inference Systems

**Recovery semantics, exactly-once token delivery, and adaptive KV-cache checkpointing for LLM serving engines (vLLM, SGLang).**

> **Status (October 2026):** Phase 1 harness built and smoke-tested; measured GPU runs pending. No results yet. A [paper draft](paper/main.pdf) has complete methods, with every results section marked pending. Every number added later will link to the raw logs and the exact command that produced it.

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

**None yet.** No measured experiments have been run. The harness has been smoke-tested on a laptop CPU to confirm it works; see the [validation note](docs/harness-validation.md). Those smoke numbers are not findings. This section will be filled only from measured runs with raw logs committed to `experiments/`.

| Phase | Gate to pass before the next phase | Status |
| --- | --- | --- |
| 1. Harness and baselines | Fault injection runs end to end on unmodified vLLM and SGLang, logs committed | Harness built and smoke-tested on CPU ([validation note](docs/harness-validation.md)); measured runs pending |
| 2. RQ1 characterization | Measured anomaly counts for naive retry and recompute | Not started |
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
│   └── harness-validation.md # smoke-test note
├── harness/                 # fault-injection harness (Phase 1)
├── gateway/                 # exactly-once token gateway (planned)
├── policies/                # fixed and adaptive checkpoint policies (planned)
├── experiments/             # raw logs from measured runs only (empty)
├── analysis/                # analyze.py (per-trial metrics) and make_tables.py (paper tables), both from raw logs
├── scripts/run_matrix.sh    # runs the full measurement grid for one engine
├── paper/                   # paper draft: methods complete, results pending
└── smoke_tests/summary.csv  # harness smoke test only, not results
```

## How to Reproduce

No measured results to reproduce yet. Setup for a laptop smoke test and for GPU runs (AWS or any Linux GPU machine) is in [`harness/README.md`](harness/README.md). The pipeline is already in place:

```bash
bash scripts/run_matrix.sh vllm          # full grid for one engine -> raw logs in experiments/
python analysis/make_tables.py           # every paper table, regenerated from experiments/
cd paper && latexmk -pdf main.tex        # paper picks up the generated tables
```

Once measured runs exist, this section will also list the exact environment (GPU, driver, CUDA, engine versions, model revision) recorded in each `run.json`.

## Related Work

- **DéjàVu** (Strati et al., ICML 2024) streams and replicates KV cache for fault-tolerant LLM serving. This project does not claim KV replication as new. It focuses on client-visible delivery semantics across failures and per-request adaptive checkpoint decisions. [Paper](https://proceedings.mlr.press/v235/strati24a.html)
- **SpotServe** (ASPLOS 2024) and **Llumnix** (OSDI 2024) migrate in-flight request state across instances, for preemption and load balancing respectively. Both assume planned moves; this project targets unplanned crashes.
- **vLLM / PagedAttention** (SOSP 2023) and **SGLang** provide block-managed KV caches, which make block-granular checkpointing feasible.
- **Rollback-recovery and stream processing:** Chandy and Lamport (1985) on consistent snapshots, Elnozahy et al. (2002) on rollback-recovery and output commit, Young (1974) and Daly (2006) on optimal checkpoint intervals, and the checkpointing and exactly-once models of Apache Flink and Spark Structured Streaming.

## Citation

A paper draft is in [`paper/`](paper/main.pdf) (methods complete, results pending; not for citation as results). To reference this repository:

```bibtex
@misc{medipally2026llmft,
  author       = {Medipally, Rajshekar},
  title        = {What Does a Streaming Client See When an {LLM} Server Crashes? A Fault-Injection Study of Client-Visible Token Anomalies in {vLLM} and {SGLang}},
  year         = {2026},
  howpublished = {\url{https://github.com/rmedipallycic/llm-inference-fault-tolerance}},
  note         = {Draft; measured results pending}
}
```

## Author

**Rajshekar Medipally**, Data Engineer and Cloud Architect, Richmond, VA
Research interests: distributed systems, fault tolerance, ML systems infrastructure
rmedipallycic@gmail.com · [github.com/rmedipallycic](https://github.com/rmedipallycic)

## License

Apache License 2.0. See [LICENSE](LICENSE).
