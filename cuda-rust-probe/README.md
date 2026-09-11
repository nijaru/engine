# CUDA Rust migration probes

Hardware-gated probes for the NVIDIA backend migration described in
[`docs/cuda-rust-migration.md`](../docs/cuda-rust-migration.md). They answer gate 1: reproducible build and
resource interoperability between Engine's current `cudarc`-based CUDA resources and the Rust CUDA stack
(`cuda-core` / `cuda-bindings` / `cuda-async`, published from NVlabs/cutile-rs) that cuTile and cuda-oxide
build on.

This directory is **not** a member of the repository workspace. These crates need the NVIDIA toolchain at
build time; `cargo test --workspace` in the repository root must keep working without it.

## Prerequisites

- NVIDIA GPU and a CUDA 13.x driver.
- CUDA toolkit 13.2 or newer, at `/usr/local/cuda-13.3` by default. `cuda-core` itself accepts 13.0+, but
  cuTile requires 13.2+ and the 4090's `sm_89` target is only present in `tileiras` from 13.2 onward.
  Override the path in `cuda-rust-probe/.cargo/config.toml` or with `CUDA_TOOLKIT_PATH`.
- Stable Rust (the crates here are edition 2024, MSRV 1.89).

## `interop`

The gate-1 ownership proof: two libraries operating over one context, one stream, and one allocation,
with exactly one deallocation owner and no host round trip between the operations.

- **Case A — `cuda-core` owns.** Context, stream, and buffer come from `cuda-core`. `cudarc`'s raw driver
  bindings, which is what Engine uses today, write directly into that allocation on that stream. `cudarc`
  allocates nothing and frees nothing.
- **Case B — `cudarc` owns.** The reverse direction, which is what a staged migration looks like:
  `cudarc` creates the context, stream, and buffer, and `cuda-core` adopts all three through its
  non-owning `borrow_with_owner` handles. The probe drops the `cuda-core` handles *before* touching the
  memory again, so a premature release fails the run instead of passing silently.

```sh
cargo run -p interop
```

For the release and leak sides of the same claim, run it under the compute sanitizer:

```sh
compute-sanitizer --tool memcheck --leak-check full target/debug/interop
```

## `tile`

The tile-track kernel proof. `cuda-core` allocates the buffer and owns the stream, cuTile wraps that allocation
with `Tensor::from_foreign` — its documented interop entry point, which holds a foreign owner alive and takes
no copy and no ownership transfer — and a cutile kernel writes into it. The probe reads the result back
through `cuda-core` and only then reclaims the owner with `Arc::try_unwrap`, so a tensor that outlived its
borrow, or an allocation freed early, fails the run instead of passing quietly.

```sh
cargo run -p tile
```

## `simt`

The SIMT-track kernel proof, and the migration's target ownership shape: `cuda-core` owns the context, the
stream, and every buffer, and an Engine-authored cuda-oxide kernel borrows them. No second allocator, no
second runtime.

This member is **its own workspace** because cuda-oxide authors kernels through a rustc codegen backend on a
pinned nightly, which the stable `interop` and `tile` crates must not inherit. cuda-oxide's own
kernel-authoring crates are unpublished, so the manifest pins the git revision; its `cargo oxide` subcommand
has to be installed once:

```sh
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git \
  --rev 26754ae52c26c097dc1c465a1e42c4c5d05a3d40 cargo-oxide
cd simt && cargo oxide doctor && cargo oxide run
```

## Evidence so far

Run on the RTX 4090 (driver 615.71.09, CUDA toolkit 13.3, sm_89) on 2026-09-10:

```text
device: NVIDIA GeForce RTX 4090 (sm_89)

case A: cuda-core owns, cudarc wrote async over the borrowed stream — ok
case B: cudarc owns, cuda-core borrowed and wrote, release order safe — ok

interop smoke: both ownership directions passed
```

```text
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

Toolchain context, all recorded in [`docs/cuda-rust-migration.md`](../docs/cuda-rust-migration.md): cuda-oxide's SIMT
track builds and runs on CUDA 13.1, cuTile needs 13.2+ before it can target `sm_8x` at all, the two tracks share
one published `cuda-core`, and `cuda-core`'s `simt` layer offers no non-owning `DeviceBuffer` — which is why
this probe demonstrates ownership transfer rather than a borrowed allocation.

Not yet covered here: an Engine-authored kernel (SIMT or tile) executing over these shared resources, and the
same smoke test wired to the project's own fixtures.
