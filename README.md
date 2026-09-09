# Ribn

A Rust-first model inference runtime and serving engine.

## Development

Use the Rust toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml).

```sh
cargo test --workspace
```

GPU execution requires NVIDIA CUDA hardware. See the [benchmark documentation](benchmarks/README.md) for model artifacts, hardware checks, and performance measurements.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Benchmarks](benchmarks/README.md)

## License

[Apache-2.0](LICENSE)
