#!/usr/bin/env bash
# Phase 3 (RQ2) grid for ONE engine on a GPU instance: the exactly-once gateway.
#
#   bash scripts/run_gateway_matrix.sh vllm
#   bash scripts/run_gateway_matrix.sh sglang     (inside the lmsysorg/sglang container, see README)
#   PARTS="overhead" bash scripts/run_gateway_matrix.sh vllm    # one part only
#
# Parts (all by default):
#   restart     1 replica; the gateway waits for the killed engine to restart.
#               T=0.0 and 0.8, kill points 10%/50%/90%, TRIALS trials each.
#   determ      as restart, T=0.0 at 50%, with the engine's deterministic mode
#               (vLLM batch invariance / SGLang deterministic inference).
#   overhead    failure-free pairs, direct vs gateway, without and with fsync.
#   failover    2 replicas, one per GPU; needs 2 GPUs (skipped otherwise).
#
# Raw logs go to experiments/<run_id>/; analysis/gateway_check.py summarizes.
set -euo pipefail

ENGINE="${1:?usage: run_gateway_matrix.sh vllm|sglang}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TRIALS="${TRIALS:-10}"
PAIRS="${PAIRS:-30}"
SEED="${SEED:-1234}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PARTS="${PARTS:-restart determ overhead failover}"
KILL_POINTS="${KILL_POINTS:-$((MAX_TOKENS / 10)) $((MAX_TOKENS / 2)) $((MAX_TOKENS * 9 / 10))}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
PY="$(command -v python || command -v python3)"   # the SGLang container has only python3

case "$ENGINE" in
  vllm)
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    CMD="env CUDA_VISIBLE_DEVICES={gpu} vllm serve $MODEL --port {port} --max-model-len $MAX_MODEL_LEN"
    DET_ENV="VLLM_BATCH_INVARIANT=1"; DET_FLAG=""
    MODE="ids" ;;
  sglang)
    # same server flags as the Phase 1 SGLang runs (scripts/sgl_grid.sh)
    CMD="env CUDA_VISIBLE_DEVICES={gpu} python3 -m sglang.launch_server --model-path $MODEL --port {port} --context-length $MAX_MODEL_LEN --mem-fraction-static 0.85"
    DET_ENV=""; DET_FLAG=" --enable-deterministic-inference"
    MODE="text" ;;   # token-id resume is only wired for vLLM's logprobs format
  *) echo "engine must be vllm or sglang" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2
  exit 1
fi
mkdir -p experiments
echo "commit $(git rev-parse HEAD), engine $ENGINE, model $MODEL, GPUs $NGPU, parts: $PARTS"

run() {   # run <label> <args...>; one failed run must not stop the grid
  echo "=== $ENGINE $1"
  shift
  if ! "$PY" -m harness.run_gateway_trial --run-type measured --engine "$ENGINE" \
       --model "$MODEL" --mode "$MODE" --max-tokens "$MAX_TOKENS" --seed "$SEED" "$@"; then
    echo "RUN FAILED: $ENGINE $*" | tee -a "experiments/failed_gateway_runs_${ENGINE}.txt"
    pkill -9 -f -- "--port 80" 2>/dev/null || true
    pkill -9 -f -- "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f -- "gateway.server" 2>/dev/null || true
    sleep 10
  fi
}

for PART in $PARTS; do
  case "$PART" in
    restart)
      for TEMP in 0.0 0.8; do
        for K in $KILL_POINTS; do
          run "restart T=$TEMP kill-at=$K" --server-cmd "$CMD" --temperature "$TEMP" \
              --kill-at "$K" --trials "$TRIALS"
        done
      done ;;
    determ)
      K=$((MAX_TOKENS / 2))
      if [ -n "$DET_ENV" ]; then export $DET_ENV; fi
      run "deterministic T=0.0 kill-at=$K" --server-cmd "$CMD$DET_FLAG" --temperature 0.0 \
          --kill-at "$K" --trials "$TRIALS"
      if [ -n "$DET_ENV" ]; then unset "${DET_ENV%%=*}"; fi ;;
    overhead)
      run "overhead" --server-cmd "$CMD" --temperature 0.0 --overhead-pairs "$PAIRS"
      run "overhead fsync" --server-cmd "$CMD" --temperature 0.0 --overhead-pairs "$PAIRS" --fsync ;;
    failover)
      if [ "$NGPU" -lt 2 ]; then echo "=== failover skipped: needs 2 GPUs, found $NGPU"; continue; fi
      for TEMP in 0.0 0.8; do
        run "failover T=$TEMP kill-at=$((MAX_TOKENS / 2))" --server-cmd "$CMD" --replicas 2 \
            --temperature "$TEMP" --kill-at "$((MAX_TOKENS / 2))" --trials "$TRIALS"
      done ;;
    *) echo "unknown part $PART" >&2; exit 2 ;;
  esac
done

"$PY" analysis/gateway_check.py --out "experiments/gateway_summary_${ENGINE}.md" || true
