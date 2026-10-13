#!/bin/bash
# Follow-up vLLM runs (host Python env with vLLM, on the A10G machine).
#   bash scripts/extras_vllm.sh smoke   # 2 quick smoke trials: batch-invariant mode, token-id recompute
#   bash scripts/extras_vllm.sh         # 26 measured runs x 5 trials, about 6 hours
# Every run's exact command line and environment is recorded in its run.json.
set -u
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
M=meta-llama/Llama-3.1-8B-Instruct
BASE="vllm serve $M --port 8000 --max-model-len 4096"
NOAPC="--no-enable-prefix-caching"

run() {  # run <type> <label> <strategy> <T> <kill> [prompt-id]
  local type=$1 label=$2 st=$3 T=$4 k=$5 pid=${6:-} sx="" envx=""
  case "$label" in
    vllm-bi)    envx="VLLM_BATCH_INVARIANT=1" ;;
    vllm-noapc) sx="$NOAPC" ;;
  esac
  local args=(--run-type "$type" --engine "$label" --server-cmd "$BASE $sx" --model "$M"
              --strategy "$st" --temperature "$T" --seed 1234 --kill-at "$k")
  if [ "$type" = smoke ]; then args+=(--max-tokens 64 --trials 1); else args+=(--max-tokens 256 --trials 5); fi
  [ -n "$pid" ] && args+=(--prompt-id "$pid")
  echo "=== $type $label ${pid:-wal} $st T=$T kill-at=$k"
  env $envx python -m harness.run_trial "${args[@]}" \
    || echo "RUN FAILED: $label ${pid:-wal} $st T=$T k=$k" | tee -a experiments/failed_runs_extras_vllm.txt
  pkill -9 -f "vllm serve" 2>/dev/null; sleep 5
}

if [ "${1:-}" = smoke ]; then
  run smoke vllm-bi naive_retry 0.0 32
  run smoke vllm recompute_ids 0.0 32
  echo "smoke done."; exit 0
fi

# 1. batch-invariant mode, original prompt
for S in naive_retry recompute; do for T in 0.0 0.8; do run measured vllm-bi $S $T 128; done; done
# 2. token-id recompute (no re-tokenization of delivered text), with and without prefix caching
run measured vllm recompute_ids 0.0 128
run measured vllm-noapc recompute_ids 0.0 128
# 3. five more prompts, default config and prefix caching off, temperature 0
for P in tcp story coffee threads crisis; do
  for L in vllm vllm-noapc; do for S in naive_retry recompute; do run measured $L $S 0.0 128 $P; done; done
done
echo "done."
