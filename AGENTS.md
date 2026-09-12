# Ribn agent guidance

Ribn (pronounced “ribbon”) is a Rust-first model inference runtime and serving
engine. The codebase is evolving quickly; current repository code, tests, and docs
are more authoritative than older handoffs or remembered architecture.

## Repository workflow

Work directly on `main` unless the maintainer asks for a branch or pull request.
Keep commits focused and leave the tree green. Public architecture/roadmap decisions
belong in `docs/`; do not make repository behavior depend on private context files.

At session start, read `README.md`, `docs/architecture.md`, and `docs/roadmap.md`,
then inspect the actual code/tests relevant to the change. Keep correctness and
performance claims tied to evidence.

## Product direction

The goal is a state-of-the-art inference engine in Rust, not a novel interaction
model. Prefer familiar local, library, and serving workflows unless Ribn has a
concrete reason to differ. Sensible defaults should reduce setup while supported
execution controls remain available.

The shared text frontend distinguishes raw prompts, structured chat, and token-ID
input. Do not introduce a user-visible task-selection system merely to mirror
internal execution categories. Likewise, do not make applications drive scheduler
steps or output-credit bookkeeping when a higher-level interface can own that work.
Lower-level APIs remain useful for embedding and advanced integration.

## Engineering direction

Treat architectural choices as hypotheses unless correctness or resource safety
makes them invariants. New models may legitimately expose missing shared concepts;
do not preserve an abstraction solely to avoid changing core code.

Useful current boundaries:

- request lifecycle and scheduling live in `crates/runtime`;
- reusable text formatting/tokenization/result handling lives in `crates/text`;
- model configuration and artifact mappings live with the model implementation;
- backend-specific resources, state layouts, and kernels stay backend/model-owned;
- protocol/CLI frontends reuse library behavior rather than reimplement generation.

These boundaries can evolve when real implementation evidence warrants it. Avoid
a lowest-common-denominator device abstraction, a universal model compiler, or a
fleet scheduler unless the project develops a concrete need for one.

## Performance and state

Keep the request hot path cheap. Prefer persistent request/device metadata,
incremental updates, asynchronous host/device execution, and backend-specific
optimized paths where measurements support them.

Do not assume all continuation state is KV. Qwen's current target combines
full-attention KV with recurrent state. Future paging/prefix reuse must preserve a
single semantic prefix across every required continuation component even if their
physical allocators/layouts differ. Dynamic resource management, preemption,
prefix caching, scheduler policy, overlap, and CUDA graphs should be designed and
measured together with the implementations they optimize rather than specified in
advance from another engine's architecture.

Optimized variants need correctness qualification for the scope in which they are
automatically selected. Compilation or host tests are not GPU evidence.

## Current first path

- Qwen3.8-27B text generation is the current model path.
- NVIDIA single-GPU execution is first; the development GPU is qualification
  hardware, not an engine-wide architectural constraint.
- GGUF is the first artifact format and comparison path, not a mandatory core
  representation.
- `ribn local` is retained only as a comparison oracle during cutover; do not add
  a second feature roadmap to the legacy runtime.

## Verification

```text
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

CUDA-feature compile checks and nonignored tests run in CPU CI. Device execution,
numerical qualification, and GPU performance measurements require a compatible
NVIDIA host. See `benchmarks/runtime-contract.md` and `docs/cuda-rust-migration.md`.
