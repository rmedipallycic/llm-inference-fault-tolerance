#!/usr/bin/env bash
# Phase 5 Option B, revision: failover kill trials with several requests in flight,
# on the FIRST machine, after
#   bash scripts/phase5b_standby.sh <same mode>
# is running on the standby. CONCURRENCY requests at once (distinct random-token
# prompts of PROMPT_LEN tokens, 256 output tokens each); replica 0 is SIGKILLed
# once request 0 has received 128 positions; every request is checked for
# delivery and for client-visible text (vs the standby's /detokenize).
#   bash scripts/run_phase5b_concurrent.sh none|prompt|all http://<standby IP>:8000
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE="${2:?usage: run_phase5b_concurrent.sh none|prompt|all http://IP:8000}"
TRIALS="${TRIALS:-3}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
CONCURRENCY="${CONCURRENCY:-4 16 32}"
# shellcheck disable=SC1091
source scripts/phase5b_common.sh "$1"
if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2; exit 1
fi
export $ENGINE_ENV
echo "commit $(git rev-parse HEAD), mode $MODE, remote $REMOTE"
for C in $CONCURRENCY; do
  echo "=== concurrent failover $MODE C=$C L=$PROMPT_LEN"
  python -m harness.run_ckpt_failover --run-type measured --label "$MODE" \
      --server-cmd "$REPLICA_CMD --port {port}" --model "$MODEL" \
      --remote "$REMOTE" --remote-desc "$REPLICA_DESC" --prompt-len "$PROMPT_LEN" \
      --kill-at 128 --trials "$TRIALS" --concurrency "$C" \
      --clear-cmd "$CLEAR_CMD" --pre-start-cmd "$PRESTART_CMD" \
    || echo "RUN FAILED: concurrent $MODE C=$C" | tee -a experiments/failed_phase5b_runs.txt
done
python analysis/phase5b.py --out experiments/phase5b_summary.md >/dev/null || true
echo "done"
