#!/bin/bash
set -u
cd /work
git config --global --add safe.directory /work
M=meta-llama/Llama-3.1-8B-Instruct
BASE="python3 -m sglang.launch_server --model-path $M --port 8000 --context-length 4096 --mem-fraction-static 0.85"
run() {
  echo "=== $1 $3 T=$4 kill-at=$5"
  python3 -m harness.run_trial --run-type measured --engine "$1" --server-cmd "$BASE $2" --model $M --strategy "$3" --temperature "$4" --seed 1234 --max-tokens 256 --kill-at "$5" --trials 5 || echo "RUN FAILED: $1 $3 T=$4 k=$5" | tee -a experiments/failed_runs_sglang.txt
  pkill -9 -f sglang.launch_server; sleep 5
}
for S in naive_retry recompute; do for T in 0.0 0.8; do for K in 25 128 230; do run sglang "" $S $T $K; done; done; done
for S in naive_retry recompute; do for T in 0.0 0.8; do run sglang-noradix "--disable-radix-cache" $S $T 128; done; done
echo "done."
