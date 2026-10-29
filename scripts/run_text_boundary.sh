#!/usr/bin/env bash
# Phase 5B follow-up, on ONE machine (no standby, no NFS): token-id prompts that end
# inside a multi-byte character, through the gateway with and without its text
# check (harness/run_text_boundary.py), then the tokenizer recheck.
#   bash scripts/run_text_boundary.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2; exit 1
fi
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
echo "commit $(git rev-parse HEAD)"
python -m harness.run_text_boundary --run-type measured --model "$MODEL" \
    --server-cmd "vllm serve $MODEL --max-model-len 4096 --port {port}" --max-prompts "${PROMPTS:-60}"
python analysis/recheck_text.py --model "$MODEL" | grep textboundary
echo "done"
