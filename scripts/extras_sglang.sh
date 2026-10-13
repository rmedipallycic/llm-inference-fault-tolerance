#!/bin/bash
# Follow-up SGLang runs. Runs INSIDE the lmsysorg/sglang container, repo mounted at /work.
#   bash /work/scripts/extras_sglang.sh smoke   # 1 quick smoke trial in deterministic mode
#   bash /work/scripts/extras_sglang.sh         # 28 measured runs x 5 trials, about 7 hours
set -u
cd /work
git config --global --add safe.directory /work
M=meta-llama/Llama-3.1-8B-Instruct
BASE="python3 -m sglang.launch_server --model-path $M --port 8000 --context-length 4096 --mem-fraction-static 0.85"

run() {  # run <type> <label> <strategy> <T> <kill> [prompt-id]
  local type=$1 label=$2 st=$3 T=$4 k=$5 pid=${6:-} sx="" extra=()
  case "$label" in
    sglang-seed) sx="--random-seed 1234" ;;
    sglang-det)  sx="--enable-deterministic-inference"; extra=(--extra-body '{"sampling_seed": 1234}') ;;
  esac
  local args=(--run-type "$type" --engine "$label" --server-cmd "$BASE $sx" --model "$M"
              --strategy "$st" --temperature "$T" --seed 1234 --kill-at "$k" "${extra[@]}")
  if [ "$type" = smoke ]; then args+=(--max-tokens 64 --trials 1); else args+=(--max-tokens 256 --trials 5); fi
  [ -n "$pid" ] && args+=(--prompt-id "$pid")
  echo "=== $type $label ${pid:-wal} $st T=$T kill-at=$k"
  python3 -m harness.run_trial "${args[@]}" \
    || echo "RUN FAILED: $label ${pid:-wal} $st T=$T k=$k" | tee -a experiments/failed_runs_extras_sglang.txt
  pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
}

if [ "${1:-}" = smoke ]; then
  run smoke sglang-det naive_retry 0.8 32
  echo "smoke done."; exit 0
fi

# 1. fixed server random seed, original prompt
for S in naive_retry recompute; do for T in 0.0 0.8; do run measured sglang-seed $S $T 128; done; done
# 2. deterministic inference mode with a sampling seed, original prompt
for S in naive_retry recompute; do for T in 0.0 0.8; do run measured sglang-det $S $T 128; done; done
# 3. five more prompts, default config and deterministic mode, temperature 0
for P in tcp story coffee threads crisis; do
  for L in sglang sglang-det; do for S in naive_retry recompute; do run measured $L $S 0.0 128 $P; done; done
done
echo "done."
