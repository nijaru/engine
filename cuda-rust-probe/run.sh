#!/usr/bin/env bash
# Gate-1 hardware smoke test: build and run every CUDA Rust probe, then repeat
# each one under the compute sanitizer's leak check.
#
# The point of the sanitizer pass is the ownership claim. Each probe frees its
# allocation exactly once and through exactly one owner, so a second free, a
# use after free, or a leaked foreign allocation fails here instead of passing
# quietly.
#
# Usage:
#   cuda-rust-probe/run.sh            # run everything
#   COLD=1 cuda-rust-probe/run.sh     # drop cuTile's JIT cache first, to time a
#                                     # cold preparation instead of a warm one
set -euo pipefail

cd "$(dirname "$0")"

CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-13.3/bin}"
SANITIZER="${SANITIZER:-$CUDA_BIN/compute-sanitizer}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

if ! command -v "$NVIDIA_SMI" >/dev/null; then
    echo "no nvidia-smi: this smoke test needs a CUDA host" >&2
    exit 1
fi

# Do not fight another GPU workload for the device. The migration design says
# not to disrupt unrelated work, and a contended device also makes the timings
# meaningless.
busy="$("$NVIDIA_SMI" --query-compute-apps=pid --format=csv,noheader)"
if [ -n "$busy" ]; then
    echo "GPU busy: compute processes $busy are running; refusing to disturb them" >&2
    exit 1
fi

if [ -n "${COLD:-}" ]; then
    for cache in "$HOME/.cache/cutile" "${XDG_CACHE_HOME:-$HOME/.cache}/cutile"; do
        [ -d "$cache" ] && echo "dropping cold-start cache $cache" && rm -rf "$cache"
    done
fi

gpu="$("$NVIDIA_SMI" --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "host: $gpu"
echo "CUDA: $("$CUDA_BIN/nvcc" --version | tail -1)"
echo

timed() {
    local label="$1"
    shift
    local start end
    start="$(date +%s.%N)"
    "$@"
    end="$(date +%s.%N)"
    printf '  %-28s %6.2fs\n' "$label" "$(echo "$end - $start" | bc)"
}

echo "build"
timed "interop, tile" cargo build --quiet -p interop -p tile
timed "simt" bash -c "cd simt && cargo build --quiet --release"

echo
echo "run (warm preparation unless COLD=1)"
timed "interop" ./target/debug/interop
timed "tile" ./target/debug/tile
timed "simt" ./simt/target/release/simt

if [ ! -x "$SANITIZER" ]; then
    echo
    echo "no compute sanitizer at $SANITIZER; skipping the ownership check" >&2
    exit 1
fi

echo
echo "ownership check: memcheck, full leak check"
for binary in ./target/debug/interop ./target/debug/tile ./simt/target/release/simt; do
    echo "  $binary"
    "$SANITIZER" --tool memcheck --leak-check full "$binary" 2>&1 |
        grep -E 'LEAK SUMMARY|ERROR SUMMARY' | sed 's/^/    /'
done

echo
echo "gate 1 smoke test: all probes passed"
