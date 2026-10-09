#!/usr/bin/env bash
# Run the full Phase 1/2 measurement grid for ONE engine on a GPU instance.
#
#   bash scripts/run_matrix.sh vllm
#   bash scripts/run_matrix.sh sglang     (in a separate Python env with sglang)
#
# Grid: 2 strategies x 2 temperatures x 3 kill points x TRIALS trials.
# Every run writes raw logs to experiments/<run_id>/ via harness/run_trial.py.
# Nothing here computes results; analysis/analyze.py does that afterwards.
set -euo pipefail

ENGINE="${1:?usage: run_matrix.sh vllm|sglang}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TRIALS="${TRIALS:-10}"
SEED="${SEED:-1234}"
PORT="${PORT:-8000}"
# Llama 3.1 defaults to a 128k context, whose KV cache does not fit a 24 GB GPU.
# Cap it; requests here use well under 1k tokens. Logged in every run.json via argv.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# 10%, 50%, 90% of MAX_TOKENS
KILL_POINTS="${KILL_POINTS:-$((MAX_TOKENS / 10)) $((MAX_TOKENS / 2)) $((MAX_TOKENS * 9 / 10))}"

case "$ENGINE" in
  vllm)   SERVER_CMD="vllm serve $MODEL --port $PORT --max-model-len $MAX_MODEL_LEN" ;;
  sglang) SERVER_CMD="python -m sglang.launch_server --model-path $MODEL --port $PORT --context-length $MAX_MODEL_LEN" ;;
  *) echo "engine must be vllm or sglang" >&2; exit 2 ;;
esac

# Provenance: refuse to run measured trials from uncommitted code.
if [ -n "$(git status --porcelain -- harness analysis scripts)" ]; then
  echo "harness/, analysis/ or scripts/ has uncommitted changes; commit first." >&2
  exit 1
fi
mkdir -p experiments
echo "commit $(git rev-parse HEAD), engine $ENGINE, model $MODEL"

for STRATEGY in naive_retry recompute; do
  for TEMP in 0.0 0.8; do
    for K in $KILL_POINTS; do
      echo "=== $ENGINE $STRATEGY T=$TEMP kill-at=$K trials=$TRIALS"
      # One failed run must not stop an unattended grid: record it and continue.
      if ! python -m harness.run_trial --run-type measured --engine "$ENGINE" \
        --server-cmd "$SERVER_CMD" --model "$MODEL" \
        --strategy "$STRATEGY" --temperature "$TEMP" --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" --kill-at "$K" --trials "$TRIALS"; then
        echo "RUN FAILED: $ENGINE $STRATEGY T=$TEMP kill-at=$K" | tee -a "experiments/failed_runs_${ENGINE}.txt"
        pkill -9 -f -- "--port $PORT" 2>/dev/null || true   # clear a half-dead server before the next run
        sleep 10
      fi
    done
  done
done

python analysis/analyze.py --csv "experiments/summary_${ENGINE}.csv" || true
if [ -s "experiments/failed_runs_${ENGINE}.txt" ]; then
  echo "SOME RUNS FAILED (kept on record):"; cat "experiments/failed_runs_${ENGINE}.txt"
fi
echo "done. Copy experiments/ off this machine, commit it, then stop the instance."
