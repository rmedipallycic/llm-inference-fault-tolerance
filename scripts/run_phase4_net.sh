#!/usr/bin/env bash
# Phase 4 (RQ3) step 2, sender side: KV-sized transfers to the standby machine.
# Start the receiver on the standby first (it serves both sessions, then keep or Ctrl-C it):
#   python -m harness.kv_net_bench receive --port 8000
# Then on this machine (vLLM venv active, no inference server running):
#   bash scripts/run_phase4_net.sh <standby private IP>
# Runs one session with 1 TCP stream and one with 4. Raw logs in experiments/<run_id>/.
set -uo pipefail
PEER="${1:?usage: run_phase4_net.sh <standby private IP>}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PY="$(command -v python || command -v python3)"
if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2
  exit 1
fi
echo "commit $(git rev-parse HEAD), peer $PEER"
for S in 1 4; do
  echo "=== streams $S"
  "$PY" -m harness.kv_net_bench send --run-type measured --peer "$PEER" --port 8000 \
      --model "$MODEL" --streams "$S" \
    || echo "RUN FAILED: kv net streams $S" | tee -a experiments/failed_phase4_runs.txt
  sleep 5
done
"$PY" analysis/phase4.py --out experiments/phase4_summary.md >/dev/null || true
echo "done; summary in experiments/phase4_summary.md"
