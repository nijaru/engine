#!/usr/bin/env bash
# Run this bounded probe serially on the qualification GPU, not in ordinary CI.
set -euo pipefail
cd "$(dirname "$0")"
# Qualification must not inherit a mode that skips checks or times sanitizers.
unset PREP_BENCH BENCH
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda-13.3}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'Refusing to run: another compute process holds the GPU.' >&2
    exit 1
fi
cargo test --locked
cargo oxide run
# Verify the packed-dot instruction was actually emitted; this alone is not
# numerical or performance evidence.
grep -Eq '^[[:space:]]*dp4a.s32.s32[[:space:]]' gate2.ptx
for tool in memcheck initcheck synccheck; do
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        echo 'Refusing sanitizer run: another compute process holds the GPU.' >&2
        exit 1
    fi
    args=(--tool "$tool" --error-exitcode 99)
    if [[ "$tool" == memcheck ]]; then args+=(--leak-check full); fi
    "$CUDA_TOOLKIT_PATH/bin/compute-sanitizer" "${args[@]}" target/release/gate2
done
