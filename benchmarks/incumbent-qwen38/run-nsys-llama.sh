#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
model=/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
tokdir=$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots" -mindepth 2 -maxdepth 2 -name tokenizer.json -printf '%h\n' | head -1)
outdir="$HOME/qwen38-bench/nsys-llama-ar-c1"
rm -rf "$outdir"
mkdir -p "$outdir"
prefix="$outdir/llama-ar-c1"

cleanup() {
  pkill -x llama-server 2>/dev/null || true
}
trap cleanup EXIT
cleanup

/usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true \
  --force-overwrite=true --stats=true \
  --output="$prefix" \
  "$HOME/.local/bin/llama-server" \
  -m "$model" -c 16384 -ngl 999 -fa on --jinja --reasoning off \
  --cache-type-k q4_0 --cache-type-v q4_0 --host 127.0.0.1 --port 8091 \
  -np 1 >"$outdir/server.log" 2>&1 &
nsys_pid=$!
ready=0
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8091/health >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "profiled llama-server did not become ready" >&2
  tail -80 "$outdir/server.log" >&2 || true
  kill "$nsys_pid" 2>/dev/null || true
  exit 1
fi
rm -rf "$outdir/aiperf"
aiperf profile \
  --model "$model" --tokenizer "$tokdir" --endpoint-type chat --streaming \
  --url http://127.0.0.1:8091 --isl 256 --isl-stddev 0 --osl 128 --osl-stddev 0 \
  --concurrency 1 --request-count 10 --num-prompts 10 --warmup-request-count 2 \
  --random-seed 42 --use-server-token-count \
  --extra-inputs '{"temperature": 0, "ignore_eos": true}' \
  --wait-for-model-timeout 30 --wait-for-model-mode models \
  --output-artifact-dir "$outdir/aiperf" --no-auto-plot
cleanup
wait "$nsys_pid" || true
ls -lh "$outdir" "$prefix"*.nsys-rep "$prefix"*.sqlite 2>/dev/null || true
