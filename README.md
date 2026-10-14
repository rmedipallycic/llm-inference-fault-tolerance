# Fault-Tolerant LLM Inference Systems

**Recovery semantics, exactly-once token delivery, and adaptive KV-cache checkpointing for LLM serving engines (vLLM, SGLang).**

> **Status (October 14, 2026):** Phase 1 measured on vLLM 0.31.0 and SGLang 0.5.21 (AWS g5.xlarge, one A10G each, Llama 3.1 8B Instruct): 86 runs, 430 trials, including cache ablations, both engines' deterministic modes, a token-id recompute check, and five further prompts. Raw logs are in [`experiments/`](experiments/), and results are in the [paper draft](paper/main.pdf) and summarized below.

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

Measured on vLLM 0.31.0 and SGLang 0.5.21 with identical harness code, one A10G each, Llama 3.1 8B Instruct, one main prompt plus five more for the follow-up runs, a 256-token budget, and requests issued one at a time. Every number comes from [`analysis/make_tables.py`](analysis/make_tables.py) run on the logs in [`experiments/`](experiments/). Scope limits are in the paper's threats-to-validity section.

1. **In both engines' default configurations, failure-free output was not unique at temperature 0, for different reasons.**
   - **vLLM:** the first request after a start and the second identical request differed in all 12 runs, and both texts were exactly repeatable. Output depended on what the server had served before.
   - **SGLang:** the second request was the same text in every run, but the first request after a start varied from start to start (10 distinct texts in 21 cold outputs).
2. **Naive retry contradicted the client.** A restarted server is cold, so the retry disagreed with text the client had already shown in **25 of 30 vLLM** and **24 of 30 SGLang** trials.
3. **Recompute never contradicted the client but changed the answer.** Its final text matched neither failure-free output of its run in 24 of 30 (vLLM) and 26 of 30 (SGLang) trials.
4. **The cache ablations split the engines.** Disabling vLLM's automatic prefix caching made output reproducible (4 of 4 runs) and removed every contradiction (0 of 10). Disabling SGLang's radix cache changed neither (contradictions in 10 of 10).
5. **Re-tokenization is not the cause.** Resubmitting token ids instead of text (vLLM) changed nothing; re-tokenizing the delivered text gave the same ids in 10 of 10 trials. SGLang's server seed is not the cause either (`--random-seed` made no difference).
6. **Deterministic modes plus recompute gave exact recovery.** With vLLM's batch invariance (`VLLM_BATCH_INVARIANT=1`) or SGLang's `--enable-deterministic-inference`, recompute reproduced the failure-free output exactly in every temperature-0 trial (35 of 35, across six prompts for SGLang and one for vLLM), and naive retry never contradicted the client. Throughput cost was not measured.
7. **The pattern holds on five more prompts.** In default configurations, naive retry contradicted the client on 3 of 5 prompts (vLLM) and 4 of 5 (SGLang).
8. **SGLang ignored the OpenAI-style seed** in its default mode, so its default 0.8 results are for unseeded sampling. Even in deterministic mode, SGLang's sampled (0.8) output was not reproducible across requests.
9. **Recovery latency was server restart.** Restart took 107 to 195 s; the first recovered token usually arrived 0.23 to 0.6 s after the server was ready (3 to 4 s in some SGLang deterministic-mode runs).

| Phase | Gate to pass before the next phase | Status |
| --- | --- | --- |
| 1. Harness and baselines | Fault injection runs end to end on unmodified vLLM and SGLang, logs committed | Done: 86 runs, 430 trials committed (SGLang via its official container; [note](docs/sglang-attempt.md)) |
| 2. RQ1 characterization | Measured anomaly counts for naive retry and recompute | Done for both engines: six prompts, cache ablations, deterministic modes ([paper](paper/main.pdf)); throughput cost of the fixes not yet measured |
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
│   └── sglang-attempt.md    # SGLang setup: failed host install, working container
├── harness/                 # fault-injection harness (Phase 1)
├── gateway/                 # exactly-once token gateway (planned)
├── policies/                # fixed and adaptive checkpoint policies (planned)
├── experiments/             # raw logs from measured runs only (86 runs, 430 trials)
├── analysis/                # analyze.py (per-trial metrics) and make_tables.py (paper tables), both from raw logs
├── scripts/run_matrix.sh    # vLLM measurement grid
├── scripts/sgl_grid.sh      # SGLang grid + radix ablation (runs inside the SGLang container)
├── scripts/extras_*.sh      # follow-up runs: deterministic modes, token-id recompute, more prompts
├── paper/                   # paper draft with measured vLLM and SGLang results
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
TRIALS=5 bash scripts/run_matrix.sh vllm # vLLM main grid: 12 runs x 5 trials
```

SGLang ran from its official container, with the harness inside the same container so that the kill reaches the server (model already in `~/.cache/huggingface`):

```bash
docker pull lmsysorg/sglang:v0.5.21      # measured with digest sha256:b1259f3e...
docker run --gpus all --rm --network host --ipc=host --shm-size 16g \
  -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD":/work -w /work \
  lmsysorg/sglang:v0.5.21 bash /work/scripts/sgl_grid.sh
```

The prefix-cache ablation used the same harness with `--engine vllm-noapc` and `--no-enable-prefix-caching` added to the server command, at `--kill-at 128` for both strategies and temperatures; the exact command line is recorded in each ablation run's `run.json`.

Recorded environment: vLLM 0.31.0 with torch 2.13.0, and SGLang 0.5.21 with torch 2.13.0+cu130 (container image `lmsysorg/sglang:v0.5.21`); NVIDIA A10G (23028 MiB), driver 595.91.07, Deep Learning OSS Nvidia Driver AMI (Ubuntu 26.04, build 20261003), Python 3.12, `VLLM_USE_FLASHINFER_SAMPLER=0` for vLLM (the host image has no CUDA toolkit). Full package versions and GPU inventory are in every `run.json`.

## Related Work

- **DéjàVu** (Strati et al., ICML 2024) streams and replicates KV cache for fault-tolerant LLM serving. This project does not claim KV replication as new. It focuses on client-visible delivery semantics across failures and per-request adaptive checkpoint decisions. [Paper](https://proceedings.mlr.press/v235/strati24a.html)
- **SpotServe** (ASPLOS 2024) and **Llumnix** (OSDI 2024) migrate in-flight request state across instances, for preemption and load balancing respectively. Both assume planned moves; this project targets unplanned crashes.
- **Defeating Nondeterminism in LLM Inference** (He et al., Thinking Machines Lab, 2025) shows that inference at temperature 0 varies with server load because kernels are not batch-invariant. The finding above is the fault-tolerance consequence: a retry after a crash is not a replay.
- **vLLM / PagedAttention** (SOSP 2023) and **SGLang** provide block-managed KV caches, which make block-granular checkpointing feasible.
- **Rollback-recovery and stream processing:** Chandy and Lamport (1985) on consistent snapshots, Elnozahy et al. (2002) on rollback-recovery and output commit, Young (1974) and Daly (2006) on optimal checkpoint intervals, and the checkpointing and exactly-once models of Apache Flink and Spark Structured Streaming.

## Citation

A paper draft with the measured vLLM and SGLang results is in [`paper/`](paper/main.pdf). To reference this repository:

```bibtex
@misc{medipally2026llmft,
  author       = {Medipally, Rajshekar},
  title        = {What Does a Streaming Client See When an {LLM} Server Crashes? A Fault-Injection Study of Client-Visible Token Anomalies in {vLLM} and {SGLang}},
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
