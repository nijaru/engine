#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-benchmarks/results/env-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out_dir"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "engine_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "engine_status=$(git status --porcelain=v1 2>/dev/null | wc -l | tr -d ' ') changed paths"
  echo
  echo "[toolchains]"
  rustc --version 2>&1 || true
  cargo --version 2>&1 || true
  python3 --version 2>&1 || true
  nvcc --version 2>&1 || true
  echo
  echo "[python-packages]"
  python3 -m pip show aiperf vllm sglang 2>/dev/null || true
  echo
  echo "[nvidia-summary]"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,pstate,power.limit,clocks.current.graphics,clocks.current.memory --format=csv,noheader 2>&1 || true
} > "$out_dir/environment.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$out_dir/nvidia-smi-q.txt" 2>&1 || true
  nvidia-smi topo -m > "$out_dir/nvidia-topology.txt" 2>&1 || true
fi

echo "$out_dir"
