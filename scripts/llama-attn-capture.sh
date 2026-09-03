#!/usr/bin/env bash
# Capture full-attention layer-3 tensors (plus the wrapper chain and the
# embedding/layer-2 input) from genuine llama.cpp inference, for the Engine
# host-reference parity test
# (crates/nvidia/tests/cuda_reference.rs
#  host_reference_full_attn_matches_llama_debug_capture).
#
# Run on the desktop against the pinned llama.cpp build 10684 checkout:
#   bash scripts/llama-attn-capture.sh
# Produces /tmp/qwen-attn-layer3-capture-{1tok,2tok}.txt.
#
# The 1-token run ("Hi", token 12675) validates the attention/FFN chain with
# rope at position 0 (identity rotation). The 2-token run ("Hi there")
# additionally exercises real rope rotation at position 1 and a 2-entry KV
# cache. Note "Kcur-3" is captured twice by llama.cpp (pre-rope and post-rope
# share the cb name); occurrences are order-significant.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/github/ggml-org/llama.cpp}"
MODEL="${MODEL:-/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf}"

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
