# Engine

A Rust-first model inference runtime and serving engine.

Early development, not for production use. The first execution target is Qwen3.8-27B on NVIDIA CUDA GPUs. The host build carries the core request/scheduler/runtime contracts, GGUF loading, and CPU references without needing a GPU.

## Requirements

- Rust 1.98 (pinned in `rust-toolchain.toml`)
- NVIDIA CUDA GPU for device execution; developed on RTX 4090-class hardware
- Pinned Qwen3.8 GGUF (~16 GB, see `benchmarks/model-set.toml`) for hardware-gated tests and benchmarks

## Quickstart

```sh
cargo test --workspace
```

Lint and CUDA feature check (match CI):

```sh
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p engine-nvidia --features cuda --all-targets -- -D warnings
cargo clippy -p engine-server --features cuda --all-targets -- -D warnings
```

Serving benchmark on CUDA (needs the pinned model):

```sh
ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  cargo run --release -p engine-nvidia --features cuda \
  --example qwen_serving_bench -- --concurrency=4 --tokens=32
```

## Layout

- `crates/core` — request, scheduler, runtime, and backend contracts
- `crates/gguf` — GGUF loading and the Qwen3.8 model description
- `crates/nvidia` — CUDA backend, execution kernels, and benchmarks
- `crates/server` — serving binary (currently a local CUDA request runner)
- `docs/` — [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md)
- `benchmarks/` — [methodology](benchmarks/README.md) and [measurement history](benchmarks/execution-history.md)

## License

[Apache-2.0](LICENSE)
