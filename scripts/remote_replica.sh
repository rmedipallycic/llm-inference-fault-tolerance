#!/usr/bin/env bash
# Start the SECOND vLLM replica for two-machine failover runs. Run this on the
# other machine (replica 1), inside tmux, and leave it running:
#
#   bash scripts/remote_replica.sh default     # default vLLM mode
#   bash scripts/remote_replica.sh bi          # batch-invariant mode
#
# It listens on all interfaces, port 8000, so the gateway on the first machine
# can reach it over the private network (security group: allow TCP 8000 from
# the same security group). The command is the same one
# scripts/run_failover_two_hosts.sh records as --remote-desc; both scripts
# take it from scripts/failover_common.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: remote_replica.sh default|bi}"
# shellcheck disable=SC1091
source scripts/failover_common.sh "$MODE"
echo "replica 1: $REPLICA_DESC"
echo "private IP for the gateway machine: $(hostname -I | awk '{print $1}')"
env $ENGINE_ENV $REPLICA_CMD --host 0.0.0.0 --port 8000 2>&1 | tee -a "$HOME/remote_replica_${MODE}.log"
