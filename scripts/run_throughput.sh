#!/usr/bin/env bash
# Phase 2 follow-up: throughput cost of the anomaly fixes, for ONE engine.
#
#   bash scripts/run_throughput.sh vllm
#   bash scripts/run_throughput.sh sglang     (inside the lmsysorg/sglang container)
#
# vLLM configs:   default, batch-invariant (VLLM_BATCH_INVARIANT=1), no-apc
# SGLang configs: default, deterministic (--enable-deterministic-inference)
# Each config is run REPS times (default 2), interleaved (A B C A B C), so a
# drift over the evening shows up as a difference between repetitions.
# Raw logs go to experiments/<run_id>/; analysis/throughput.py summarizes.
set -uo pipefail

ENGINE="${1:?usage: run_throughput.sh vllm|sglang}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONC="${CONC:-1,4,16,32}"
REPS="${REPS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PY="$(command -v python || command -v python3)"

if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2
  exit 1
fi
mkdir -p experiments
echo "commit $(git rev-parse HEAD), engine $ENGINE, concurrency $CONC, reps $REPS"

case "$ENGINE" in
  vllm)
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # as in Phases 1 and 3
    BASE="vllm serve $MODEL --port 8000 --max-model-len $MAX_MODEL_LEN"
    CONFIGS="default batch-invariant no-apc" ;;
  sglang)
    BASE="python3 -m sglang.launch_server --model-path $MODEL --port 8000 --context-length $MAX_MODEL_LEN --mem-fraction-static 0.85"
    CONFIGS="default deterministic" ;;
  *) echo "engine must be vllm or sglang" >&2; exit 2 ;;
esac

run() {   # run <label> <server cmd>
  echo "=== $ENGINE $1"
  if ! "$PY" -m harness.run_throughput --run-type measured --engine "$ENGINE" --label "$1" \
       --server-cmd "$2" --model "$MODEL" --concurrency "$CONC"; then
    echo "RUN FAILED: $ENGINE $1" | tee -a "experiments/failed_throughput_runs_${ENGINE}.txt"
  fi
  pkill -9 -f -- "--port 8000" 2>/dev/null || true
  pkill -9 -f -- "sglang.launch_server" 2>/dev/null || true
  sleep 10
}

for REP in $(seq 1 "$REPS"); do
  for CFG in $CONFIGS; do
    case "$CFG" in
      default)         run default "$BASE" ;;
      batch-invariant) VLLM_BATCH_INVARIANT=1 run batch-invariant "$BASE" ;;
      no-apc)          run no-apc "$BASE --no-enable-prefix-caching" ;;
      deterministic)   run deterministic "$BASE --enable-deterministic-inference" ;;
    esac
  done
done

"$PY" analysis/throughput.py --out experiments/throughput_summary.md >/dev/null || true
echo "done; summary in experiments/throughput_summary.md"
