#!/usr/bin/env bash
# Record full environment baseline for the Qwen3.8-27B DFlash2 investigation.
set -euo pipefail
OUT="${1:-env-baseline.txt}"
{
  echo "== date"; date -Is
  echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm,pcie.link.gen.max --format=csv
  echo "== cuda"; nvcc --version 2>/dev/null | tail -1 || true
  echo "== llama.cpp"; cd ~/github/ggml-org/llama.cpp && git log --oneline -1 && git status -sb | head -1
  echo "== build flags"; grep -E "GGML_CUDA|GGML_FFMPEG|CUDA_ARCHITECTURES|GGML_NATIVE" build/CMakeCache.txt | grep -v ADVANCED
  echo "== binaries"; ~/github/ggml-org/llama.cpp/build/bin/llama-server --version 2>&1 | head -2
  echo "== dflash2 build (if present)"; ls -d ~/github/ggml-org/llama.cpp/build-dflash2 2>/dev/null && (cd ~/github/ggml-org/llama.cpp/build-dflash2 && ../build-dflash2/bin/llama-server --version 2>&1 | head -2) || true
  echo "== models"; ls -la ~/models/qwen38-27b ~/models/dflash2 ~/models/qwen38-dflash2 2>/dev/null
  echo "== vram now"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
} | tee "$OUT"
