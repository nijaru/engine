# Engine

Engine is a Rust-first inference execution and serving engine.

The project is in early development. The initial goal is a competitive single-GPU NVIDIA inference path with low host overhead, state-aware execution, hardware-specialized backends, and runtime-configurable performance policy.

## Build

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
