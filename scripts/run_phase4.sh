#!/usr/bin/env bash
# Phase 4 (RQ3), first step on ONE GPU machine (vLLM venv active):
#   1. recompute side: prefill time against context length (harness/run_prefill.py),
#      prefix caching off, then on as a check (fresh prompts mean no cache hits either way)
#   2. checkpoint side: raw KV copy bandwidth GPU <-> host <-> disk (harness/kv_copy_bench.py)
#
#   bash scripts/run_phase4.sh
#
# Raw logs go to experiments/<run_id>/; analysis/phase4.py summarizes.
set -uo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LENGTHS="${LENGTHS:-512,1024,2048,4096,8192,16384,32768}"
REPS="${REPS:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-33000}"   # 32768-token prompt + 1 output token
PY="$(command -v python || command -v python3)"

if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2
  exit 1
fi
mkdir -p experiments
echo "commit $(git rev-parse HEAD), model $MODEL, lengths $LENGTHS, reps $REPS"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
BASE="vllm serve $MODEL --port 8000 --max-model-len $MAX_MODEL_LEN"

for CFG in noapc apc; do
  EXTRA=""; [ "$CFG" = noapc ] && EXTRA=" --no-enable-prefix-caching"
  echo "=== prefill $CFG"
  "$PY" -m harness.run_prefill --run-type measured --engine vllm --label "$CFG" \
      --server-cmd "$BASE$EXTRA" --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" \
    || echo "RUN FAILED: prefill $CFG" | tee -a experiments/failed_phase4_runs.txt
  pkill -9 -f -- "--port 8000" 2>/dev/null || true
  sleep 10
done

echo "=== kv copy"
"$PY" -m harness.kv_copy_bench --run-type measured --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" \
  || echo "RUN FAILED: kv copy" | tee -a experiments/failed_phase4_runs.txt

"$PY" analysis/phase4.py --out experiments/phase4_summary.md >/dev/null || true
echo "done; summary in experiments/phase4_summary.md"
