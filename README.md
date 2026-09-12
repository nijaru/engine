# Ribn

A Rust-first model inference runtime and serving engine.

The goal is state-of-the-art inference performance, model support, and
reliability in Rust, with familiar CLI, HTTP, and library interfaces. Sensible
defaults should reduce setup, not remove useful configuration or advanced access.
This is the product direction, not a claim that the current implementation has
reached those goals. See the [interface plan](docs/ground-up-design.md#public-interface-direction).

The model-neutral runtime is the `ribn` crate in `crates/runtime`. Its
`GenerationExecutor` boundary keeps model state, artifact formats, and device execution
out of the common request scheduler. The first real adapter is Qwen GGUF on CUDA.

## Status

The existing Qwen path has correctness and performance evidence recorded in
[execution history](benchmarks/execution-history.md). The new runtime and Qwen
adapter have host contract tests and CUDA-feature compilation checks; their
integration is **experimental pending GPU qualification**. New model families,
HTTP serving, multimodal input, and automatic model selection are not implemented.
The generation contract is an internal execution boundary, not a requirement for
users to choose a task or executor before running a model.

The binary is `ribn`; the package is `ribn-cli`. Metadata inspection needs no GPU:

```sh
cargo run -p ribn-cli -- inspect /path/model.gguf
```

## Development

Use the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

NVIDIA execution requires compatible hardware and runtime libraries:

```sh
cargo run -p ribn-cli --features cuda -- run \
  --model /path/model.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

`run` streams through the new runtime. `local` remains the legacy correctness
frontend during migration. Neither command is a general multi-architecture
model loader yet.

## Code map

| Location | Responsibility |
| --- | --- |
| `crates/runtime` (`ribn`) | Model-neutral request ownership, scheduling, output, and generation-executor contract |
| `crates/qwen` | Qwen definition, GGUF mapping, and CUDA generation executor |
| `crates/nvidia` | NVIDIA physical state, resources, and kernels |
| `crates/gguf` | Generic GGUF reader/tokenizer; model interpretation belongs to the model package |
| `crates/core` | Legacy execution/state/runtime contracts retained for cutover and reference tests |
| `crates/cli` | Experimental streaming CLI and legacy local frontend |

## Documentation

- [Ground-up target and implementation gaps](docs/ground-up-design.md)
- [Architecture](docs/architecture.md)
- [Runtime redesign and research](docs/runtime-redesign.md)
- [Roadmap and retirement gates](docs/roadmap.md)
- [Runtime verification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
