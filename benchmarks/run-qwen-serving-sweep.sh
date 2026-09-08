#!/usr/bin/env bash
set -euo pipefail

model="${ENGINE_QWEN_GGUF:-}"
if [[ -z "$model" ]]; then
  echo "set ENGINE_QWEN_GGUF to the pinned Qwen3.8 GGUF" >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  echo "ENGINE_QWEN_GGUF does not exist: $model" >&2
  exit 2
fi

# GPU gate: never compete with a running workload for VRAM.
used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$used_mib" -gt 2000 ]] || [[ "$compute_apps" -ne 0 ]]; then
  echo "GPU busy (used=${used_mib}MiB, compute_apps=${compute_apps}); stop the running workload first" >&2
  exit 3
fi

tokens="${TOKENS:-32}"
concurrencies="${CONCURRENCIES:-1 2 4 8}"
gemv="${GEMV:-scalar}"
case "$gemv" in
  scalar|warp|int-dot) ;;
  *) echo "GEMV must be scalar, warp, or int-dot, got: $gemv" >&2; exit 2 ;;
esac
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-benchmarks/results/${stamp}-engine-qwen38-serving-sweep-gemv-${gemv}}"
mkdir -p "$out_dir/runs"

benchmarks/collect-env.sh "$out_dir/env" >/dev/null

{
  echo "model=$model"
  echo "tokens=$tokens"
  echo "concurrencies=$concurrencies"
  echo "gemv=$gemv"
  echo "engine_commit=$(git rev-parse HEAD)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out_dir/sweep.txt"

cargo build --release -p engine-nvidia --features cuda --example qwen_serving_bench 
bench="target/release/examples/qwen_serving_bench"

for concurrency in $concurrencies; do
  log="$out_dir/runs/concurrency-${concurrency}.log"
  echo "== concurrency=$concurrency tokens=$tokens ==" | tee "$log"
  "$bench" \
    --model="$model" \
    --concurrency="$concurrency" \
    --tokens="$tokens" \
    --gemv="$gemv" 2>&1 | tee -a "$log"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_dir/sweep.txt"
printf '%s\n' "$out_dir"
