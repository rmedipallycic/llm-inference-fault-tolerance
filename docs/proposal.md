# Fault-Tolerant LLM Inference: Exactly-Once Token Delivery and Adaptive KV-Cache Checkpointing

Rajshekar Medipally · October 2026

> **Status: research proposal, not a results paper.** As of October 2026, no code has been written and no experiments have been run. Every section below describes planned work. The Results section stays empty until measured data, with preserved raw logs, exists.

## Abstract

Large language model (LLM) inference servers stream tokens to clients over requests that can last tens of seconds, while holding gigabytes of per-request key-value (KV) cache in accelerator memory. When a worker fails mid-generation, today's systems mostly restart the request from the prompt, discarding the KV cache and any partially streamed output. This proposal studies two questions that the stream processing literature has answered for dataflow systems but not for LLM serving. First, what delivery guarantee does a streaming client actually receive across a worker failure: are tokens lost, duplicated, or replaced by a divergent continuation? Second, when is it worth checkpointing KV cache state rather than recomputing it, and can an adaptive policy choose per request better than any static one? The planned work is a fault-injection characterization study on two open-source serving engines (vLLM and SGLang), a token-level exactly-once delivery protocol, and an adaptive KV checkpoint policy evaluated against recompute-only and fixed-interval baselines.

## 1. Introduction

LLM serving has become stateful streaming infrastructure, but its failure handling is still mostly stateless. A generation request holds a growing KV cache on one or more accelerators and emits tokens to a client incrementally. If the process, GPU, or node fails, the usual recovery is to resubmit the prompt and generate again.

That recovery has two costs that are rarely measured together. The first is performance: the prefill must be recomputed, and every token decoded so far must be decoded again, so recovery time grows with both prompt length and output length. The second is correctness at the client. A client that has already received 400 tokens may receive them again, receive a gap, or receive a different continuation, because sampling with temperature above zero is not reproducible across a restart unless the random state is preserved.

The stream processing community has spent two decades on exactly this pair of problems: how often to checkpoint operator state, how to recover it, and how to give downstream consumers exactly-once output. My industry work on Spark Structured Streaming and Kafka pipelines, and my earlier simulation study of checkpoint interval selection, sit in that tradition. This project asks how much of that toolkit transfers to LLM inference, where the state is a large, append-only tensor and the output is a token stream read by a person or an agent.

The planned contributions are:

1. A fault-injection characterization of how two open-source serving engines (vLLM and SGLang) behave when a worker fails mid-generation, measuring recovery latency, wasted compute, and client-visible token anomalies.
2. A definition of exactly-once token delivery for streamed generation, and a lightweight protocol (sequence-numbered tokens, a persisted token log, and preserved sampler state) that provides it.
3. An adaptive KV checkpoint policy that decides per request whether to checkpoint, and how often, based on context length, tokens generated, and observed failure rate, evaluated against recompute-only and fixed-interval baselines.

## 2. Background

**Autoregressive serving.** A request runs in two phases. Prefill processes the whole prompt in parallel and builds the KV cache. Decode then generates one token per step, appending one entry per layer to the cache and streaming the token to the client. Modern engines batch many requests together and manage the cache in fixed-size blocks.

**Why the KV cache is the state that matters.** The cache is deterministic given the model weights and the token sequence, so it can always be rebuilt. Rebuilding it, however, costs a full prefill over prompt plus generated tokens. For long contexts this is the dominant recovery cost. Copying the cache out, on the other hand, costs memory bandwidth, network, and storage proportional to its size.

**Rollback-recovery.** Classic distributed systems give the vocabulary used here: checkpointing versus logging, the optimal checkpoint interval as a function of checkpoint cost and failure rate, and output commit, the rule that a message to the outside world must not be released until the state that produced it is recoverable. Stream processors such as Flink and Spark Structured Streaming apply these ideas with periodic snapshots plus replayable sources and idempotent or transactional sinks.

**The mapping this project tests.** The KV cache plays the role of operator state. The token sequence plays the role of a replayable input log, since replaying it through prefill reconstructs the cache. The streaming client plays the role of the external sink, which is where exactly-once delivery is decided.

## 3. Related work and positioning

The novelty here is narrow, and the proposal should say so. KV-cache replication for fault tolerance already exists. What is not yet characterized is client-visible delivery semantics across failures and per-request adaptive checkpoint decisions.

**Fault-tolerant LLM serving.** [DéjàVu](https://proceedings.mlr.press/v235/strati24a.html) (Strati et al., ICML 2024) is the closest prior work. It builds a KV cache streaming library and uses it for prompt-token disaggregation, microbatch swapping, and state replication for fault tolerance, implemented on FasterTransformer ([code](https://github.com/msr-fiddle/dejavu)). This proposal does not claim KV replication as new. It differs in three ways: it measures what the streaming client observes, not only server-side recovery time; it treats replication frequency as a per-request decision rather than a fixed policy; and it targets a block-managed engine rather than FasterTransformer.

**Preemption and migration.** SpotServe (Miao et al., ASPLOS 2024) serves on preemptible instances and migrates in-flight context when instances are reclaimed. Llumnix (Sun et al., OSDI 2024) live-migrates requests across instances for load balancing. Both move KV state deliberately, with warning, and are the natural baselines for the planned-migration case. Unplanned crashes without warning are the focus here.

**Serving engines.** Orca (Yu et al., OSDI 2022) introduced iteration-level scheduling. vLLM (Kwon et al., SOSP 2023) introduced block-based KV memory management, which is what makes partial, block-granular checkpointing plausible.

**Checkpointing and exactly-once in stream processing.** Chandy and Lamport (1985) define consistent snapshots. Elnozahy et al. (2002) survey rollback-recovery, including output commit. Young (1974) and Daly (2006) derive optimal checkpoint intervals from checkpoint cost and mean time to failure. Carbone et al. describe Flink's asynchronous barrier snapshots, and Armbrust et al. (SIGMOD 2018) describe Spark Structured Streaming's end-to-end exactly-once model. These supply the formal baseline the adaptive policy will be compared against.

## 4. Research questions and hypotheses

Each hypothesis is stated so that a measured result could refute it.

| ID | Research question | Hypothesis (to test, not assumed) |
| --- | --- | --- |
| RQ1 | What does a streaming client observe when a serving worker fails mid-generation under default recovery? | With sampling temperature above zero, a naive retry produces duplicated or divergent tokens in a substantial share of failed requests. |
| RQ2 | What does exactly-once token delivery cost? | Sequence numbers, a token log, and saved sampler state give exactly-once delivery with small per-token overhead, since the log grows by one token per step. |
| RQ3 | When does checkpointing KV state beat recomputing it? | A crossover exists: recompute wins for short contexts, checkpointing wins past some context length that depends on hardware and failure rate. |
| RQ4 | Does an adaptive per-request policy beat fixed policies? | An adaptive policy matches or beats the best fixed interval across mixed workloads, because no single interval suits both short chat turns and long-context requests. |

If RQ3 finds no crossover within realistic context lengths on the test hardware, RQ4 loses its motivation. That outcome is still a publishable negative result and would be reported as such.

## 5. Proposed approach

**Failure model.** Fail-stop failures of a serving worker process: process kill, GPU reset, and node loss. The gateway and a small durable store (token log and checkpoint metadata) are assumed to survive. Byzantine faults and silent data corruption are out of scope.

**Recovery strategies compared.**

1. **Recompute (baseline):** resubmit prompt plus already-delivered tokens, rebuild the KV cache by prefill, continue decoding.
2. **Naive retry (baseline):** resubmit the original prompt only, which is what many client libraries do.
3. **Fixed-interval KV checkpoint:** copy new KV blocks to host memory or a peer every k tokens; on failure, restore the last checkpoint and recompute only the tail.
4. **Adaptive KV checkpoint (proposed):** decide per request, at each block boundary, whether to checkpoint.

**Exactly-once token delivery.** Each token carries a request id and a sequence number. The gateway appends tokens to a durable log before releasing them to the client, which is the output-commit rule from rollback-recovery. Sampler state (seed and RNG position) is logged with the token. On recovery, the replacement worker resumes from the last committed sequence number, and the gateway drops any token whose sequence number it has already released. The client therefore sees each position exactly once and never sees a divergent continuation.

**Adaptive checkpoint rule.** The starting point is the classical first-order optimal interval, where C is checkpoint cost and M is mean time to failure:

```math
T_{opt} \approx \sqrt{2 \, C \, M}
```

The classical rule assumes constant checkpoint cost. Here, both checkpoint cost (proportional to new KV bytes since the last checkpoint) and recompute cost (prefill over the full sequence) grow as a request runs. The proposed policy checkpoints when the expected recompute saved over the next block exceeds the cost of the checkpoint:

```math
\lambda \, \Delta t \cdot R(n) > C(n - n_{ckpt})
```

where λ is the observed failure rate, Δt is the time to decode the next block, R(n) is the cost to recompute a sequence of n tokens, and C is the cost to copy the KV entries added since the last checkpoint at position n_ckpt. R and C will be measured, not assumed, during the RQ3 microbenchmarks.

## 6. Experimental methodology

**Setup (planned).** Two open-source serving engines with block-managed KV caches, vLLM and SGLang, each serving a small open-weight model (roughly 1B to 8B parameters) on a single cloud GPU instance. A small model keeps cost low and is enough to measure the recompute-versus-checkpoint crossover. A multi-GPU tensor-parallel setup is a later stage, only if budget allows.

**Fault injection.** A harness sends a request, waits until token position p, then kills the worker process with SIGKILL. Kill positions are drawn from a fixed list (for example 10%, 50%, and 90% of the target output length) so results are reproducible. Every trial records the kill timestamp, the server log, and the full token stream the client received.

**Workloads.** Short chat turns (short prompt, short output) and long-context requests (long prompt, long output), plus a mixed trace. Temperature 0 and temperature above 0 are run separately, since they behave differently on retry.

**Metrics.**

| Metric | Unit | Answers |
| --- | --- | --- |
| Recovery latency: kill to next token delivered | ms | RQ3, RQ4 |
| Wasted compute: tokens decoded more than once | tokens | RQ3, RQ4 |
| Checkpoint overhead on failure-free requests | % slowdown in time per output token | RQ2, RQ3 |
| Client anomalies: duplicated, missing, or divergent tokens | count per failed request | RQ1, RQ2 |
| Goodput under an injected failure rate | requests/s meeting a latency target | RQ4 |

**Provenance rules.** These are lessons from the August 2026 audit of my earlier simulation study, and they are non-negotiable for this project:

- Every number in any paper or README traces to a raw log file committed to the repository, with the exact command and commit hash that produced it.
- No hardcoded outcome parameters. Simulation, if used at all, appears in a separate, clearly labeled section and is never mixed with measured results.
- Statistics are computed by one script directly from the raw logs, re-runnable by anyone.
- Planned experiments are never described in past tense.

## 7. Results

No results yet. This section will be written only from measured runs whose raw logs are committed to the repository. Until then, it stays empty on purpose.

## 8. Threats to validity and plan

**Threats to validity.**

- **Scale.** A small model on one GPU may not show the same crossover as a 70B model under tensor parallelism. Results will be scoped to the tested configuration.
- **Failure realism.** SIGKILL is a clean fail-stop. Real GPU faults can hang rather than crash, which this harness does not model.
- **Determinism.** Even with saved sampler state, GPU kernels may not be bitwise deterministic across a restart, so the protocol may still see divergence. Measuring this is part of RQ1, not an assumption.
- **Engine internals.** Checkpointing KV blocks requires hooks into the engine. If that proves too invasive, the fallback is host-memory swap mechanisms the engine already provides.

**Phased plan.** Each phase has a gate that must be met before the next begins.

1. **Harness and baselines.** Gate: fault-injection harness runs end to end against an unmodified engine, with raw logs committed.
2. **RQ1 characterization.** Gate: measured anomaly counts for naive retry and recompute, at temperature 0 and above 0.
3. **Exactly-once protocol (RQ2).** Gate: zero client anomalies across the injected trials, with measured overhead.
4. **Checkpoint microbenchmarks (RQ3).** Gate: measured R(n) and C(n) curves on the test hardware.
5. **Adaptive policy (RQ4).** Gate: comparison against all baselines on the mixed workload.
6. **Write-up.** Only after phases 1 to 5 have measured data.

## References

1. F. Strati, S. McAllister, A. Phanishayee, J. Tarnawski, A. Klimovic. DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving. *ICML 2024*, PMLR 235, pp. 46745–46771. [Link](https://proceedings.mlr.press/v235/strati24a.html)
2. X. Miao, C. Shi, J. Duan, X. Xi, D. Lin, B. Cui, Z. Jia. SpotServe: Serving Generative Large Language Models on Preemptible Instances. *ASPLOS 2024*, pp. 1112–1127. [arXiv:2311.15566](https://arxiv.org/abs/2311.15566)
3. B. Sun, Z. Huang, H. Zhao, W. Xiao, X. Zhang, Y. Li, W. Lin. Llumnix: Dynamic Scheduling for Large Language Model Serving. *OSDI 2024*. [Link](https://www.usenix.org/conference/osdi24/presentation/sun-biao)
4. G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, B.-G. Chun. Orca: A Distributed Serving System for Transformer-Based Generative Models. *OSDI 2022*, pp. 521–538. [Link](https://www.usenix.org/conference/osdi22/presentation/yu)
5. W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, I. Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
6. K. M. Chandy, L. Lamport. Distributed Snapshots: Determining Global States of Distributed Systems. *ACM Transactions on Computer Systems* 3(1), pp. 63–75, 1985. [doi:10.1145/214451.214456](https://doi.org/10.1145/214451.214456)
7. E. N. Elnozahy, L. Alvisi, Y.-M. Wang, D. B. Johnson. A Survey of Rollback-Recovery Protocols in Message-Passing Systems. *ACM Computing Surveys* 34(3), pp. 375–408, 2002. [doi:10.1145/568522.568525](https://doi.org/10.1145/568522.568525)
8. J. W. Young. A First Order Approximation to the Optimum Checkpoint Interval. *Communications of the ACM* 17(9), pp. 530–531, 1974.
9. J. T. Daly. A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps. *Future Generation Computer Systems* 22(3), pp. 303–312, 2006.
10. P. Carbone, G. Fóra, S. Ewen, S. Haridi, K. Tzoumas. Lightweight Asynchronous Snapshots for Distributed Dataflows. arXiv:1506.08603, 2015. [Link](https://arxiv.org/abs/1506.08603)
11. M. Armbrust, T. Das, J. Torres, B. Yavuz, S. Zhu, R. Xin, A. Ghodsi, I. Stoica, M. Zaharia. Structured Streaming: A Declarative API for Real-Time Applications in Apache Spark. *SIGMOD 2018*, pp. 601–613.
