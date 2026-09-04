#!/usr/bin/env bash
# Capture full-attention layer-3 tensors from genuine llama.cpp inference for
# the Engine host-reference parity test.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/github/ggml-org/llama.cpp}"
MODEL="${MODEL:-$HOME/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf}"

FILTER='model.input_embed$|attn_norm-0$|l_out-2$|attn_norm-3$|Qcur_full-3$|Qcur_normed-3$|Kcur_normed-3$|Vcur-3$|Qcur-3$|Kcur-3$|gate_reshaped-3$|gate_sigmoid-3$|attn_pregate-3$|attn_gated-3$|attn_output-3$|attn_residual-3$|attn_post_norm-3$|ffn_out-3$|post_ffn-3$|l_out-3$'

for variant in 1tok 2tok; do
  case "$variant" in
    1tok) PROMPT="Hi" ;;
    2tok) PROMPT="Hi there" ;;
  esac
  OUT="/tmp/qwen-attn-layer3-capture-$variant.txt"
  "$LLAMA_CPP/build/bin/llama-debug" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -ngl 0 -t 1 -c 512 --no-warmup \
    --tensor-filter "$FILTER" \
    > "$OUT" 2>&1
  echo "capture written to $OUT"
done
