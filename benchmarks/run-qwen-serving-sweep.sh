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
workload="${WORKLOAD:-}"
queue_capacity="${QUEUED_REQUESTS:-0}"
if [[ -n "$workload" ]]; then
  [[ -f "$workload" ]] || { echo "WORKLOAD does not exist: $workload" >&2; exit 2; }
  gemv="${GEMV:-warp}"
else
  gemv="${GEMV:-scalar}"
fi
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
  echo "workload=$workload"
  echo "queue_capacity=$queue_capacity"
  echo "continuation_capacity_bytes=${CONTINUATION_CAPACITY_BYTES:-auto}"
  echo "ttft_slo_ms=${TTFT_SLO_MS:-unset}"
  echo "itl_slo_ms=${ITL_SLO_MS:-unset}"
  echo "e2e_slo_ms=${E2E_SLO_MS:-unset}"
  if [[ -n "$workload" ]]; then sha256sum "$workload"; fi
  echo "engine_commit=$(git rev-parse HEAD)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out_dir/sweep.txt"

cargo build --release -p engine-qwen --features cuda --example qwen_serving_bench
bench="target/release/examples/qwen_serving_bench"

for concurrency in $concurrencies; do
  log="$out_dir/runs/concurrency-${concurrency}.log"
  echo "== concurrency=$concurrency tokens=$tokens ==" | tee "$log"
  args=(--model="$model" --concurrency="$concurrency" --gemv="$gemv")
  if [[ -n "$workload" ]]; then
    args+=(--workload="$workload" --queue-capacity="$queue_capacity" --result-json="$out_dir/runs/concurrency-${concurrency}.json")
    [[ -z "${TTFT_SLO_MS:-}" ]] || args+=(--ttft-slo-ms="$TTFT_SLO_MS")
    [[ -z "${ITL_SLO_MS:-}" ]] || args+=(--itl-slo-ms="$ITL_SLO_MS")
    [[ -z "${E2E_SLO_MS:-}" ]] || args+=(--e2e-slo-ms="$E2E_SLO_MS")
  else
    args+=(--tokens="$tokens")
  fi
  [[ -z "${CONTINUATION_CAPACITY_BYTES:-}" ]] || args+=(--continuation-capacity-bytes="$CONTINUATION_CAPACITY_BYTES")
  "$bench" "${args[@]}" 2>&1 | tee -a "$log"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_dir/sweep.txt"
printf '%s\n' "$out_dir"
