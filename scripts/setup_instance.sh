#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu GPU machine, then a 1-trial pilot.
#
#   export HF_TOKEN=...            # Hugging Face Read token with Llama 3.1 access
#   bash scripts/setup_instance.sh vllm      (or: sglang)
#
# The pilot is a SMOKE run (written to smoke_tests/, never reported). It only
# checks that the server starts, the kill/restart works, and logs are written.
set -euo pipefail

ENGINE="${1:?usage: setup_instance.sh vllm|sglang}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VENV="$HOME/venv-$ENGINE"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Run:  read -s HF_TOKEN && export HF_TOKEN" >&2; exit 1
fi
command -v tmux >/dev/null || sudo apt-get install -y tmux

echo "== Python environment $VENV"
python3 -m venv "$VENV" 2>/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q python3-venv && python3 -m venv "$VENV"; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
case "$ENGINE" in
  vllm)   python -m pip install -q vllm ;;
  sglang) python -m pip install -q sglang || python -m pip install -q "sglang[all]" ;;
  *) echo "engine must be vllm or sglang" >&2; exit 2 ;;
esac

echo "== Downloading $MODEL (checks Llama access)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','tokenizer*'])"

case "$ENGINE" in
  vllm)   SERVER_CMD="vllm serve $MODEL --port 8000 --max-model-len $MAX_MODEL_LEN" ;;
  sglang) SERVER_CMD="python -m sglang.launch_server --model-path $MODEL --port 8000 --context-length $MAX_MODEL_LEN" ;;
esac

echo "== Pilot: 1 smoke trial (server start, kill, restart, recovery)"
python -m harness.run_trial --run-type smoke --engine "$ENGINE" \
  --server-cmd "$SERVER_CMD" --model "$MODEL" \
  --strategy naive_retry --temperature 0.0 --seed 1234 \
  --max-tokens 64 --kill-at 32 --trials 1
python analysis/analyze.py --smoke

cat <<MSG

== Setup and pilot OK. Start the measured grid inside tmux so it survives disconnects:

  tmux new -s grid
  source $VENV/bin/activate && export HF_TOKEN
  bash scripts/run_matrix.sh $ENGINE 2>&1 | tee grid_$ENGINE.log

Detach with Ctrl-b then d. Reattach later with: tmux attach -t grid
MSG
