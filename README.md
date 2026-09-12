# Ribn

A Rust-first model inference runtime and serving engine.

The model-neutral runtime is the `ribn` crate in `crates/runtime`. Its
`PreparedModel` boundary keeps model state, artifact formats, and device execution
out of the common request scheduler. The first real adapter is Qwen GGUF on CUDA.

## Status

The existing Qwen path has correctness and performance evidence recorded in
[execution history](benchmarks/execution-history.md). The new runtime and Qwen
adapter have host contract tests and CUDA-feature compilation checks; their
integration is **experimental pending GPU qualification**. New model families,
HTTP serving, multimodal input, and automatic model selection are not implemented.

## Development

Use the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

NVIDIA execution requires compatible hardware and runtime libraries:

```sh
cargo run -p engine-server --features cuda -- run \
  --model /path/model.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

`run` streams through the new runtime. `local` remains the legacy correctness
frontend during migration. Neither command is a general multi-architecture
model loader yet.

## Code map

| Location | Responsibility |
| --- | --- |
| `crates/runtime` (`ribn`) | Model-neutral request ownership, scheduling, output, and prepared-model contract |
| `crates/qwen` | Qwen preparation and transitional adapter to the existing CUDA executor |
| `crates/nvidia` | NVIDIA physical state, resources, and kernels |
| `crates/gguf` | Existing GGUF reader/tokenizer and Qwen metadata adapter |
| `crates/core` | Legacy execution/state/runtime contracts retained for cutover and reference tests |
| `crates/server` | Experimental streaming CLI and legacy local frontend |

## Documentation

- [Architecture](docs/architecture.md)
- [Runtime redesign and research](docs/runtime-redesign.md)
- [Roadmap and retirement gates](docs/roadmap.md)
- [Runtime verification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
