#!/usr/bin/env bash
# Phase 5 Option B smoke test, on the FIRST machine (NFS mounted, standby not needed):
# does the OffloadingConnector write blocks to /srv/kv, and does a NEW server process
# load them instead of recomputing? Starts vLLM twice with the same 8,192-token
# prompt and compares time to first token.
#   bash scripts/phase5b_smoke.sh [prompt|all]
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/phase5b_common.sh "${1:-all}"
export $ENGINE_ENV
eval "$CLEAR_CMD"
eval "$PRESTART_CMD"
run_once() {
  eval "$REPLICA_CMD --port 8000" > "$HOME/phase5b_smoke_$1.log" 2>&1 &
  for i in $(seq 1 300); do curl -s localhost:8000/v1/models >/dev/null && break; sleep 2; done
  python - <<'PY'
import random, time, requests
ids = [random.Random("smoke").randrange(1000, 120000) for _ in range(8192)]
t0 = time.time()
r = requests.post("http://localhost:8000/v1/completions", json={
    "model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": ids, "max_tokens": 64,
    "temperature": 0, "ignore_eos": True}, timeout=600)
print("status", r.status_code, "seconds", round(time.time() - t0, 2))
time.sleep(5)
m = requests.get("http://localhost:8000/metrics").text
print("\n".join(l for l in m.splitlines() if l.startswith("vllm:kv_offload") and "_bucket" not in l))
PY
  pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 10   # the engine is a separate process
  eval "$PRESTART_CMD"
}
echo "=== first server (stores)";  run_once 1
echo "files in $KV_DIR: $(find "$KV_DIR" -type f | wc -l), size $(du -sh "$KV_DIR" | cut -f1)"
echo "=== second server (should load)"; run_once 2
grep -i -E "error|offload" "$HOME"/phase5b_smoke_*.log | grep -v -i "warning" | tail -20
