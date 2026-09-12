# Ribn

A Rust-first model inference runtime and serving engine.

The goal is state-of-the-art inference performance, model support, reliability,
and familiar CLI/library/server workflows. The current implementation is still an
experimental Qwen GGUF/CUDA path under qualification; the documentation separates
implemented behavior from planned interfaces and performance work.

## Current interfaces

Metadata inspection requires no GPU:

```sh
cargo run -p ribn-cli -- inspect /path/model.gguf
```

The experimental CUDA path supports one-shot text generation. Text is treated as
a user chat message by default; `--raw` performs raw completion instead. The model
can be positional or supplied by the existing `--model` form.

```sh
cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --prompt 'Explain a mutex.' --max-tokens 128

cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --raw --prompt 'The answer is'
```

File input and piped stdin are supported. Interactive terminal chat and an HTTP
`serve` command are not implemented yet. `ribn local` remains the legacy numerical
comparison frontend during migration.

## Library layers

`crates/runtime` (`ribn`) is the low-level generation runtime. `crates/text`
(`ribn-text`) is the shared text frontend used by the CLI: it provides raw prompt,
chat-message, and token-ID input, tokenization/chat-template handling, incremental
UTF-8 decoding, synchronous streaming, offline batching, and terminal token usage.
The current high-level stream borrows the model mutably; a concurrent application
handle/server driver is planned rather than implied.

The first real model implementation is Qwen on CUDA. Qwen configuration is
separate from GGUF parsing, while NVIDIA resources and kernels remain backend-owned.

## Development

Use the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

CUDA-feature compile/tests run in CPU CI, but numerical/lifecycle qualification
requires an actual compatible NVIDIA GPU.

## Code map

| Location | Responsibility |
| --- | --- |
| `crates/runtime` (`ribn`) | Token-generation request lifecycle, scheduling, output, and executor contract |
| `crates/text` (`ribn-text`) | Shared text/chat/token input and generation-result frontend |
| `crates/qwen` | Qwen definition, GGUF mapping, and current CUDA executor |
| `crates/nvidia` | NVIDIA physical state, resources, and kernels |
| `crates/gguf` | Generic GGUF metadata/tensor reader and tokenizer support |
| `crates/core` | Legacy execution/state contracts retained for cutover/reference tests |
| `crates/cli` | `ribn` command-line frontend |

## Documentation

- [Architecture](docs/architecture.md)
- [Ground-up design and external lessons](docs/ground-up-design.md)
- [Roadmap](docs/roadmap.md)
- [Runtime redesign history](docs/runtime-redesign.md)
- [Runtime verification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
