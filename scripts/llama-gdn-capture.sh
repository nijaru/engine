#!/usr/bin/env bash
# Capture layer-0 Gated-DeltaNet tensors from genuine llama.cpp inference
# for the Engine host-reference parity test
# (crates/nvidia/tests/cuda_reference.rs host_reference_gdn_matches_llama_debug_capture).
#
# Run on the desktop against the pinned llama.cpp build 10684 checkout:
#   bash scripts/llama-gdn-capture.sh
# Produces /tmp/qwen-gdn-layer0-capture.txt.
#
# The prompt "Hi" is a single token (12675, no BOS for this model), which
# forces the autoregressive GDN path with a zero initial state on a fresh
# context, matching the Rust test's zero-state fixture. Single-threaded CPU
# execution keeps the reference deterministic.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/github/ggml-org/llama.cpp}"
MODEL="${MODEL:-/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf}"
OUT="${OUT:-/tmp/qwen-gdn-layer0-capture.txt}"

"$LLAMA_CPP/build/bin/llama-debug" \
  -m "$MODEL" \
  -p "Hi" \
  -ngl 0 -t 1 -c 512 --no-warmup \
  --tensor-filter 'model.input_embed$' \
  --tensor-filter 'attn_norm-0$' \
  --tensor-filter 'linear_attn_qkv_mixed-0$' \
  --tensor-filter 'z-0$' \
  --tensor-filter 'beta_sigmoid-0$' \
  --tensor-filter 'gate-0$' \
  --tensor-filter 'conv_states-0$' \
  --tensor-filter 'conv_output_raw-0$' \
  --tensor-filter 'conv_output_silu-0$' \
  --tensor-filter 'q_conv_predelta-0$' \
  --tensor-filter 'k_conv_predelta-0$' \
  --tensor-filter 'v_conv_predelta-0$' \
  --tensor-filter 'attn_output-0$' \
  --tensor-filter 'final_output-0$' \
  --tensor-filter 'linear_attn_out-0$' \
  > "$OUT" 2>&1

echo "capture written to $OUT"
