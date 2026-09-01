# Engine

Engine is a Rust-first model inference runtime and serving engine.

It is intended to cover the core inference-engine layer: model loading and execution, quantized/local inference, server inference, batching and scheduling, inference-state/cache management, hardware backends and kernels, profiling/runtime policy, and eventually multi-GPU and distributed inference.

The project is in early development. The first target is a competitive single-GPU NVIDIA path for Qwen3.8-27B, with low host overhead, correct hybrid inference-state handling, hardware-specialized execution, and live runtime-configurable performance policy.

## Build

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
