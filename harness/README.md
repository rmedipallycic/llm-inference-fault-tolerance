# Phase 1: fault-injection harness

Starts an OpenAI-compatible inference server, records failure-free reference
outputs, then kills the server mid-generation (SIGKILL of the whole process
tree), restarts it, applies a client-side recovery strategy, and saves every
raw token chunk with timestamps to JSON.

**Two kinds of runs, never mixed:**

| `--run-type` | Written to | Use |
| --- | --- | --- |
| `smoke` | `smoke_tests/` (git-ignored) | Checking the harness works, e.g. on a laptop. Never reported. |
| `measured` | `experiments/` (committed) | Real runs on the target setup. The only source of reported numbers. |

## Install

```bash
python -m pip install -r requirements.txt
```

## Laptop smoke test (Windows, no GPU needed)

vLLM and SGLang need Linux with an NVIDIA GPU, so on a laptop use
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, which
exposes the same OpenAI-compatible API, with any small GGUF model.

```bash
python -m harness.run_trial --run-type smoke --engine llamacpp \
  --server-cmd "llama-server -m path/to/small-model.gguf --port 8000" \
  --model local --strategy naive_retry --max-tokens 128 --kill-at 40 --trials 2
python analysis/analyze.py --smoke
```

## Measured runs (AWS GPU instance)

1. Check your EC2 quota for "Running On-Demand G and VT instances" in your
   region before anything else. New accounts often start at 0, and a quota
   increase can take a day or more.
2. Launch a GPU instance with at least 24 GB of GPU memory (for example
   g5.xlarge, one A10G) from an AWS Deep Learning AMI for Ubuntu.
3. On the instance:
   ```bash
   git clone https://github.com/rmedipallycic/llm-inference-fault-tolerance.git
   cd llm-inference-fault-tolerance
   python -m pip install -r requirements.txt vllm
   ```
   Llama 3.1 8B is gated on Hugging Face: accept the license on the model
   page, then run `huggingface-cli login` on the instance.
4. Run, for example:
   ```bash
   python -m harness.run_trial --run-type measured --engine vllm \
     --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000" \
     --model meta-llama/Llama-3.1-8B-Instruct \
     --strategy naive_retry --temperature 0.8 --seed 1234 \
     --max-tokens 256 --kill-at 128 --trials 10
   python analysis/analyze.py --csv experiments/summary.csv
   ```
5. Commit everything in `experiments/` (raw JSON plus `server.log`) together
   with the code version that produced it.
6. **Stop the instance when finished.** GPU instances bill by the hour while running.

## What each trial records

`run.json` holds the config, git commit, GPU, and two failure-free reference
outputs (and whether they were identical). Each `trial_NNN.json` holds:

- `chunks_at_kill`: chunks received when the kill was triggered
- `in_flight_chunks_after_kill`: chunks the server had already sent, which
  still reached the client after it died
- `pre_kill_chunks` / `recovery_chunks`: every chunk with wall and monotonic timestamps
- `restart_s`: time for the server to come back and answer health checks
- `recovery_latency_s`: kill to first token delivered after recovery
- `client_view_text`: everything the client received, in order

## Known limits

- One chunk is usually one token in vLLM and SGLang, but not always, so
  chunk counts are approximate token counts. Analysis compares text.
- Recovery latency includes a full server restart, since this phase runs a
  single instance. Failover to a standby instance is later work.
- SIGKILL is a clean fail-stop; hangs and GPU errors are not modeled.
