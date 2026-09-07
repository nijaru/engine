# Engine

Engine is a Rust-first model inference runtime and serving engine.

The project is in early development. Its goal is to cover the inference-engine layer: model loading and execution, quantized/local inference, request scheduling and continuous batching, typed inference-state management, hardware-specialized backends and execution variants, runtime policy/profiling, and eventually multi-GPU and distributed inference.

The first implementation target is the Qwen3.8-27B text path on NVIDIA. The native CUDA correctness path has a greedy 200-token parity gate against a pinned llama.cpp reference. The same execution path is now connected to persistent request slots, deterministic token-budget scheduling, typed logical and physical inference state, explicit submission/completion ownership, cancellation/reclamation, committed generated-token delivery, and a direct local Qwen command. The NVIDIA dispatcher executes compatible decode rows through native batched kernels and uses the single-row path for other work. Competitive serving performance and network serving remain unfinished.

Engine is experimental and not production-ready.

## Design

Two principles shape the current architecture:

- inference state is typed and model-defined rather than assumed to be only a KV cache;
- the steady-state runtime should avoid rebuilding request metadata, unnecessary allocation, and CPU/GPU synchronization on every step.

The Qwen hybrid path keeps full-attention KV and recurrent/Gated-DeltaNet state as separate state families. Recurrent state describes its actual persistent storage geometry rather than reusing model projection-head dimensions. Logical state identity remains in core while CUDA owns the physical allocations.

The RTX 4090 is the first testable target in the RTX 3090-and-up class; broader common hardware support is a long-term goal. GGUF is the first local artifact format, not a required core representation. Hardware-specific implementations are expected to use the best available vendor, external, JIT/AOT, fused, graph, or custom paths when they are correct and measured.

See [`docs/architecture.md`](docs/architecture.md) for the runtime boundaries and [`docs/roadmap.md`](docs/roadmap.md) for implementation order.

## Workspace

- `crates/core` — model/request/state/execution/backend contracts and shared runtime semantics
- `crates/gguf` — GGUF parsing, tokenizer metadata, quantized tensor access, and the first Qwen provider
- `crates/nvidia` — CUDA/NVIDIA execution, physical state, Qwen execution, and CUDA reference tests
- `crates/server` — direct/local frontend now; streaming/network serving follows over the same runtime
- `benchmarks` — reproducible model/workload/reference metadata; local result output is ignored

## Build and verify

Rust 1.98 is pinned by `rust-toolchain.toml`.

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

CUDA feature composition is also checked by CI without requiring a GPU. Actual CUDA execution and performance qualification require a compatible NVIDIA host and the pinned model artifact.

The current direct Qwen correctness frontend can be built and run on such a host with:

```text
cargo run --release -p engine-server --features cuda -- \
  local --model /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --max-tokens 8
```

This command is a qualification/front-end baseline, not a throughput claim.

## Benchmark policy

Performance claims must record the model artifact/revision, quantization, workload semantics, hardware/runtime, and relevant runtime policy. Correctness-qualified eager paths are the reference before graph/capture or speculative variants are allowed to become automatic choices.

Current benchmark metadata lives under [`benchmarks/`](benchmarks/). Large model artifacts and benchmark results are not committed.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
