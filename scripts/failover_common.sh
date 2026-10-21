#!/usr/bin/env bash
# Shared definition of the vLLM replica for two-machine failover runs.
# Sourced by scripts/remote_replica.sh and scripts/run_failover_two_hosts.sh,
# so that both machines run the same engine command and settings.
MODE="${1:?mode: default|bi}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
REPLICA_CMD="vllm serve $MODEL --max-model-len $MAX_MODEL_LEN"
case "$MODE" in
  default) ENGINE_ENV="VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1" ;;
  bi)      ENGINE_ENV="VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1" ;;
  *) echo "mode must be default or bi" >&2; return 2 2>/dev/null || exit 2 ;;
esac
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
REPLICA_DESC="vllm $VLLM_VERSION, mode $MODE, env: $ENGINE_ENV, cmd: $REPLICA_CMD --host 0.0.0.0 --port 8000"
