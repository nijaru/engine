# Ribn

A Rust-first model inference runtime and serving engine.

Early development, not for production use.

Ribn (pronounced “ribbon”) is the working project name; repository and crate identifiers still use `engine`.

## Requirements

- Rust (see `rust-toolchain.toml`)
- NVIDIA CUDA GPU for device execution
- Pinned model artifact for hardware-gated tests and benchmarks (see `benchmarks/model-set.toml`)

## Quickstart

```sh
cargo test --workspace
```

GPU benchmarks and serving sweeps are documented under [`benchmarks/`](benchmarks/README.md).

## Docs

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Benchmarks](benchmarks/)

## License

[Apache-2.0](LICENSE)
