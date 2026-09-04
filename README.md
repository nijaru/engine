# Engine

Engine is a Rust-first model inference runtime and serving engine.

The project is in early development. Its goal is to cover the inference-engine layer: model loading and execution, quantized/local inference, request scheduling and continuous batching, typed inference-state management, hardware-specialized backends and execution variants, runtime policy/profiling, and eventually multi-GPU and distributed inference.

The first implementation target is the Qwen3.8-27B text path on NVIDIA. A native single-request path now executes end to end and has a greedy 200-token parity gate against a pinned llama.cpp reference. The next phase is the serving/runtime foundation: generic state boundaries, persistent request slots, async-first execution, continuous batching, state paging/reuse, and a minimal server over the same runtime.

Engine is experimental and not production-ready.

## Design

Two principles shape the current architecture:

- inference state is typed and model-defined rather than assumed to be only a KV cache;
- the steady-state runtime should avoid rebuilding request metadata, unnecessary allocation, and CPU/GPU synchronization on every step.

The RTX 4090 is development and qualification hardware, not an architectural target. GGUF is the first local artifact format, not a required core representation. Hardware-specific implementations are expected to use the best available vendor, external, JIT/AOT, fused, graph, or custom paths when they are correct and measured.

See [`docs/architecture.md`](docs/architecture.md) for the runtime boundaries and [`docs/roadmap.md`](docs/roadmap.md) for implementation order.

## Workspace

- `crates/core` — model/request/state/execution/backend contracts and shared runtime semantics
- `crates/gguf` — GGUF parsing, tokenizer metadata, quantized tensor access, and the first Qwen provider
- `crates/nvidia` — CUDA/NVIDIA execution, physical state, Qwen execution, and CUDA reference tests
- `crates/server` — serving/frontend work as the runtime grows into Phase 4
- `benchmarks` — reproducible model/workload/reference metadata; local result output is ignored

## Build and verify

Rust 1.98 is pinned by `rust-toolchain.toml`.

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

CUDA-specific checks require a compatible NVIDIA/CUDA host and are intentionally separate from the portable workspace CI.

## Benchmark policy

Performance claims must record the model artifact/revision, quantization, workload semantics, hardware/runtime, and relevant runtime policy. Correctness-qualified eager paths are the reference before graph/capture or speculative variants are allowed to become automatic choices.

Current benchmark metadata lives under [`benchmarks/`](benchmarks/). Large model artifacts and benchmark results are not committed.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
