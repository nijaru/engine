#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
tokdir=$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots" -mindepth 2 -maxdepth 2 -name tokenizer.json -printf '%h\n' | head -1)
[ -n "$tokdir" ] || { echo "tokenizer snapshot not found" >&2; exit 1; }
out="$HOME/qwen38-bench/aiperf-llama-isl256-osl128-c1"
rm -rf "$out"
exec aiperf profile \
  --model /home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --tokenizer "$tokdir" \
  --endpoint-type chat \
  --streaming \
  --url http://127.0.0.1:8091 \
  --isl 256 --isl-stddev 0 \
  --osl 128 --osl-stddev 0 \
  --concurrency 1 \
  --request-count 10 \
  --num-prompts 10 \
  --warmup-request-count 2 \
  --random-seed 42 \
  --use-server-token-count \
  --extra-inputs '{"temperature": 0, "ignore_eos": true}' \
  --wait-for-model-timeout 30 --wait-for-model-mode models \
  --output-artifact-dir "$out" \
  --no-auto-plot
