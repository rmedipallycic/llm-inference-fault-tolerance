#!/usr/bin/env bash
# Phase 5 Option B: failover kill trials, on the FIRST machine, after
#   bash scripts/phase5b_standby.sh <same mode>
# is running on the standby. Prompt lengths 4,096 and 16,384, kill after 128
# output positions, TRIALS trials each.
#   bash scripts/run_phase5b_failover.sh none|prompt|all http://<standby IP>:8000
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE="${2:?usage: run_phase5b_failover.sh none|prompt|all http://IP:8000}"
TRIALS="${TRIALS:-5}"
LENGTHS="${LENGTHS:-4096 16384}"
# shellcheck disable=SC1091
source scripts/phase5b_common.sh "$1"
if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2; exit 1
fi
export $ENGINE_ENV
echo "commit $(git rev-parse HEAD), mode $MODE, remote $REMOTE"
for L in $LENGTHS; do
  echo "=== failover $MODE L=$L"
  python -m harness.run_ckpt_failover --run-type measured --label "$MODE" \
      --server-cmd "$REPLICA_CMD --port {port}" --model "$MODEL" \
      --remote "$REMOTE" --remote-desc "$REPLICA_DESC" --prompt-len "$L" \
      --kill-at 128 --trials "$TRIALS" --clear-cmd "$CLEAR_CMD" \
    || echo "RUN FAILED: failover $MODE L=$L" | tee -a experiments/failed_phase5b_runs.txt
done
python analysis/phase5b.py --out experiments/phase5b_summary.md >/dev/null || true
echo "done"
