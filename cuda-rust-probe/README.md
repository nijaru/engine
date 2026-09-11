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

## Evidence so far

Recorded in [`docs/cuda-rust-migration.md`](../docs/cuda-rust-migration.md) (2026-09-10): cuda-oxide's SIMT
track builds and runs on CUDA 13.1, cuTile requires 13.2+ for `sm_8x`, the two tracks share one published
`cuda-core`, and `cuda-core`'s `simt` layer offers no non-owning `DeviceBuffer`, which is why the probe
demonstrates ownership transfer rather than a borrowed allocation.
