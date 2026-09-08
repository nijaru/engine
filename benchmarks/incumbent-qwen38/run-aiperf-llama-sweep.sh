#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
model=/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
tokdir=$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots" -mindepth 2 -maxdepth 2 -name tokenizer.json -printf '%h\n' | head -1)
[ -n "$tokdir" ] || { echo "tokenizer snapshot not found" >&2; exit 1; }
base="$HOME/qwen38-bench/aiperf-llama-sweep-isl256-osl128"
mkdir -p "$base"

cleanup() {
  pkill -x llama-server 2>/dev/null || true
}
trap cleanup EXIT

for c in 2 4 8; do
  cleanup
  rm -rf "$base/c$c"
  nohup "$HOME/.local/bin/llama-server" \
    -m "$model" -c 16384 -ngl 999 -fa on --jinja --reasoning off \
    --cache-type-k q4_0 --cache-type-v q4_0 --host 127.0.0.1 --port 8091 \
    -np "$c" >"$base/server-c$c.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8091/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "llama-server did not become ready for concurrency $c" >&2
    tail -40 "$base/server-c$c.log" >&2 || true
    kill "$server_pid" 2>/dev/null || true
    exit 1
  fi
  aiperf profile \
    --model "$model" --tokenizer "$tokdir" --endpoint-type chat --streaming \
    --url http://127.0.0.1:8091 --isl 256 --isl-stddev 0 --osl 128 --osl-stddev 0 \
    --concurrency "$c" --request-count 16 --num-prompts 16 --warmup-request-count 4 \
    --random-seed 42 --use-server-token-count \
    --extra-inputs '{"temperature": 0, "ignore_eos": true}' \
    --wait-for-model-timeout 30 --wait-for-model-mode models \
    --output-artifact-dir "$base/c$c" --no-auto-plot
  cleanup
  sleep 2
done
