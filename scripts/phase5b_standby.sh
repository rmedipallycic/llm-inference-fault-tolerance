#!/usr/bin/env bash
# Phase 5 Option B, on the STANDBY machine (inside tmux): replica 1 with the given
# offload mode, reading and writing the local RAM disk /srv/kv.
#   bash scripts/phase5b_standby.sh none|prompt|all
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/phase5b_common.sh "$1"
echo "standby: $REPLICA_DESC"
eval "$PRESTART_CMD"
eval "env $ENGINE_ENV $REPLICA_CMD --host 0.0.0.0 --port 8000" 2>&1 | tee -a "$HOME/phase5b_standby_${MODE}.log"
