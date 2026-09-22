# Representative kernel probe (gate 2 candidate deferred)

This isolated cuda-oxide executable is **not a production backend** and does not
close migration gate 2. The current candidate is deferred because matched single-row
projection and GDN timings regress against C++; the migration document owns the
verdict and re-entry conditions. The production backend remains unchanged.
Use the pinned nightly and upstream revision in this directory; the checked-in lock
file pins transitive dependencies. Toolkit selection inherits `../.cargo/config.toml`.

On an idle qualification GPU:

```sh
./run.sh                         # host/device checks, memcheck/initcheck/synccheck
BENCH=1 cargo oxide run           # packing/projection/GDN paired timings
PREP_BENCH=pack target/release/gate2       # module + entrypoint preparation
PREP_BENCH=projection target/release/gate2
PREP_BENCH=state target/release/gate2
cargo clippy --locked --all-targets -- -D warnings
```

`run.sh` refuses to start while another compute process is present. Reserve the GPU
operationally as well: this occupancy check is not an exclusive lock. `BENCH=1` alone
does not check occupancy. Timings use CUDA events across 100 launches per sample;
they exclude preparation and transfers but can include host launch gaps. GDN timing
adds 700 identical-input updates on each side, followed by another C++ bit comparison;
these extra updates are not included in the independent f64 qualification.

Preparation modes create one context pair and print eight fresh-wrapper construction
times (first plus seven repeats). Run each mode in seven fresh processes, both with
`CUDA_CACHE_DISABLE=1` and with populated default caches. Context creation and Rust AOT
build time are excluded; C++ compilation/loading is included. The state constructors
have different bundle scopes, so this is not whole-model startup comparison. Like
`BENCH`, preparation modes require an explicit occupancy check. `run.sh` clears both
modes to ensure qualification cannot be skipped.

Implemented checks:

- Exact Q8_1 words versus scalar host quantization and the existing C++ packer:
  1, 3, 4, 5, 8, 129, 160, 480 and 1280 blocks; zero blocks, signed half-integer ties, differing
  scales and partial final CTAs. Host assertions cover empty/partial/u32-overflow
  input geometry and literal tie-rounding bytes. Finite input and half-representable
  headers remain preconditions, not device-side rejection claims.
- A real **device-to-device** packing → projection chain on one stream: Q4_K
  144-byte blocks, Q8_1 36-byte blocks, signed `dp4a`, all six-bit scale/min groups.
  `(K,N)` = `(256,1)`, `(768,5)`, `(4096,128)`, `(5120,5120)` at M=1,3,8.
  Rust output must match C++ f32 bits and a separate scalar f64 packed-dot result.
  The f64 bound is the existing Q4/Q8 test's magnitude × eps × 32 + 1e-5.
  A second independent check compares the original float-input projection against
  its input-derived quantization-error bound plus that rounding allowance. Packed
  device words are also checked separately, so compensating errors cannot pass.
- Batched GDN with eight **separately allocated** pointer slots, not a single
  contiguous state tensor. Active `(M,VH,KH,D)` = `(1,2,1,16)`, `(3,4,2,32)`,
  `(8,4,2,128)`, `(3,48,16,128)`, `(8,48,16,128)`. The `(3,4,2,32)` case runs
  64 updates; others run eight. Inputs vary each step, q/k are normalized, gates
  include zero/one endpoints, and initial states include zero and nonzero matrices.
  Live allocation slots swap on odd steps; inactive pads remain unchanged.
  Every state/output element must match C++ bits and a row-major f64 recurrence.
  That reference propagates input-derived f32 rounding envelopes through the entire
  history, without resets from device results. It permits ordinary FMA contraction,
  not arbitrary reassociation, overflow or subnormal operand flushing. Its rsqrt
  allowance follows the PTX specification, not observed error. Host tests cover
  hand-calculated recurrence, head/member/offset indexing and fused cancellation.
  These synthetic histories are not full-model numerical qualification or cuTile
  partitioning/performance evidence.

Numerical comparisons check finiteness. The binary exits unsuccessfully on mismatch;
memcheck uses a nonzero error exit code. Kernels are private probe implementation,
called with constructed valid buffers, not general safe launch APIs. Do not reuse
these entrypoints in production without whole-shape/context validation and the
existing execution owner's retirement contract.

## Remaining gate requirements

- Full malformed-buffer, member-count, geometry and context rejection tests, with
  unchanged output sentinels. Current geometry assertions only cover packing sizes.
- Resolve the measured kernel regressions before reopening promotion. Packing/GDN,
  specialized C++ single-row and preparation timings now have recorded evidence.
- Actual loaded-machine-code inspection. Offline sm_89 ptxas/SASS inspection confirms
  integer dots and local loads/stores;
  it is not inspection of the actual driver-JIT image or measured occupancy.
- Gate 3 completion/cancellation/uncertain-enqueue ownership tests before integration.

Numerical agreement on these fixtures is not full-model qualification or evidence
that CUDA Rust improves serving performance. See `../../docs/cuda-rust-migration.md`.
