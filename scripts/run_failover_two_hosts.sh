#!/usr/bin/env bash
# Phase 3 failover across two machines. Run on the FIRST machine (gateway +
# replica 0), after scripts/remote_replica.sh <same mode> is running on the
# second machine:
#
#   bash scripts/run_failover_two_hosts.sh default http://<private IP of machine 2>:8000
#   bash scripts/run_failover_two_hosts.sh bi      http://<private IP of machine 2>:8000
#
# default: T=0.0 and T=0.8, kill at 128, TRIALS trials each.
# bi:      T=0.0, kill at 128, TRIALS trials (batch-invariant on both machines).
# In every trial the gateway starts on local replica 0, which is SIGKILLed at
# position 128; the gateway resumes on the remote replica. Replica 0 restarts
# in the background before the next trial.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: run_failover_two_hosts.sh default|bi http://IP:8000}"
REMOTE="${2:?remote replica URL, e.g. http://172.31.5.10:8000}"
TRIALS="${TRIALS:-10}"
# shellcheck disable=SC1091
source scripts/failover_common.sh "$MODE"

if [ -n "$(git status --porcelain -- harness analysis scripts gateway)" ]; then
  echo "harness/, analysis/, scripts/ or gateway/ has uncommitted changes; commit first." >&2
  exit 1
fi
remote_version="$(curl -s --max-time 5 "$REMOTE/version" | python -c 'import json,sys; print(json.load(sys.stdin).get("version"))' || true)"
if [ "$remote_version" != "$VLLM_VERSION" ]; then
  echo "remote replica at $REMOTE reports vLLM '$remote_version', local is '$VLLM_VERSION' (or it is not reachable)." >&2
  echo "Start it with: bash scripts/remote_replica.sh $MODE   (and allow TCP 8000 in the security group)" >&2
  exit 1
fi
export $ENGINE_ENV
echo "commit $(git rev-parse HEAD), mode $MODE, remote $REMOTE ($REPLICA_DESC)"

temps="0.0 0.8"
[ "$MODE" = bi ] && temps="0.0"
for T in $temps; do
  echo "=== failover $MODE T=$T kill-at=128"
  python -m harness.run_gateway_trial --run-type measured --engine "vllm" \
    --server-cmd "$REPLICA_CMD --port {port}" --model "$MODEL" --mode ids \
    --remote-backends "$REMOTE" --remote-desc "$REPLICA_DESC" \
    --max-tokens 256 --seed 1234 --temperature "$T" --kill-at 128 --trials "$TRIALS" \
    || echo "RUN FAILED: failover $MODE T=$T" | tee -a experiments/failed_gateway_runs_vllm.txt
done
python analysis/gateway_check.py --out experiments/gateway_summary_vllm.md || true
