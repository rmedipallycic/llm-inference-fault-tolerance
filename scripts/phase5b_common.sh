#!/usr/bin/env bash
# Shared settings for Phase 5 Option B (sourced by the phase5b scripts).
# MODE: none   - no KV connector (recompute on the standby, as in Phase 3)
#       prompt - vLLM OffloadingConnector, prompt KV only (offload_prompt_only)
#       all    - OffloadingConnector, prompt and every generated block of 16 tokens
# Both machines use the same shared directory path, KV_DIR: on the standby it is
# a RAM disk exported over NFS; on the first machine it is the NFS mount.
MODE="${1:?mode: none|prompt|all}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-17000}"
KV_DIR="${KV_DIR:-/srv/kv}"
CPU_BYTES="${CPU_BYTES:-4294967296}"     # 4 GiB CPU tier, required by the connector
ENGINE_ENV="VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1"
case "$MODE" in
  none)   KV_ARGS="" ;;
  prompt) PO=true ;;
  all)    PO=false ;;
  *) echo "mode must be none, prompt or all" >&2; return 2 2>/dev/null || exit 2 ;;
esac
if [ "$MODE" != none ]; then
  KV_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU_BYTES,\"offload_prompt_only\":$PO,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_DIR\"}]}}"
  KV_ARGS="--kv-transfer-config '$KV_JSON'"
fi
REPLICA_CMD="vllm serve $MODEL --max-model-len $MAX_MODEL_LEN $KV_ARGS"
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
REPLICA_DESC="vllm $VLLM_VERSION, offload $MODE, env: $ENGINE_ENV, cmd: $REPLICA_CMD --host 0.0.0.0 --port 8000"
CLEAR_CMD="find $KV_DIR -type f -name '*.bin*' -delete"   # block files only; keep config.json and dirs
