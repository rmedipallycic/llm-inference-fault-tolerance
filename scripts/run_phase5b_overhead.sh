#!/usr/bin/env bash
# Phase 5 Option B: decoding cost of offloading, on the FIRST machine (NFS mounted;
# the standby replica is not needed). For each mode, a fresh server, 4,096-token
# prompts (distinct per request), 256 output tokens, 1 and 4 requests in flight.
#   bash scripts/run_phase5b_overhead.sh
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${REPS:-2}"
if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2; exit 1
fi
mkdir -p experiments
echo "commit $(git rev-parse HEAD)"
for REP in $(seq 1 "$REPS"); do
  for M in none prompt all; do
    # shellcheck disable=SC1091
    source scripts/phase5b_common.sh "$M"
    echo "=== overhead $M (rep $REP)"
    env $ENGINE_ENV python -m harness.run_throughput --run-type measured --engine vllm \
        --label "offload-$M" --server-cmd "$REPLICA_CMD --port 8000" --model "$MODEL" \
        --concurrency 1,4 --requests-per-level 8,16 --random-prompt-len 4096 \
        --clear-cmd "$CLEAR_CMD" --pre-start-cmd "$PRESTART_CMD" \
      || echo "RUN FAILED: overhead $M" | tee -a experiments/failed_phase5b_runs.txt
    pkill -9 -f -- "--port 8000" 2>/dev/null || true
    sleep 10
  done
done
python analysis/phase5b.py --out experiments/phase5b_summary.md >/dev/null || true
echo "done"
