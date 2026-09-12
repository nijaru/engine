# Ribn agent guidance

Ribn (pronounced “ribbon”) is the working name for this Rust-first model inference runtime and serving engine. Most repository and backend identifiers still use `engine`; `crates/runtime` now exports the model-neutral `ribn` crate.

## Repository workflow

Work directly on `main` unless the maintainer asks for a branch or pull request. Keep commits narrow and leave the tree green.

If maintainer-local companion context is available, use it for private planning and handoff details, but do not make repository behavior depend on unavailable private files. Public architecture and roadmap decisions belong in `docs/`.

## Session start

1. Read `README.md`.
2. Read `docs/architecture.md` and `docs/roadmap.md` for work that changes runtime boundaries or sequencing.
3. Inspect the code and tests relevant to the task.
4. Check repository status before editing.
5. Keep implementation and performance claims evidence-backed.

## Runtime migration

New request/runtime work belongs in `crates/runtime`, through `PreparedModel`.
Keep concrete model state, layouts, artifact formats, and kernel mechanisms in
model/backend implementations. Do not add model-family variants to the new
scheduler. `crates/qwen` currently bridges the existing CUDA executor; the new
`run` path remains experimental until hardware qualification. `local` and legacy
core serving types remain comparison oracles, not a second feature target.
Read `docs/runtime-redesign.md` and the cutover gates before changing this boundary.

## Current first target

- Qwen3.8-27B, text path first.
- NVIDIA single-GPU execution is the first implementation path; the RTX 4090 is development and qualification hardware, not an architectural target.
- The first artifact is a Q4 GGUF used for same-artifact parity against llama.cpp. GGUF is a loader concern, not a required core representation.
- Qwen3.8 requires recurrent/linear-attention state plus full-attention KV state. Do not design a KV-only runtime boundary.

## Architecture guardrails

- Engine owns inference execution: model loading/execution, quantized/local inference, serving, scheduling/batching, inference state, backends/kernels, profiling/runtime policy, and eventually distributed inference.
- Engine must remain independently deployable under bare metal, containers, Kubernetes, Slurm, or another orchestrator.
- External orchestrators own physical resource allocation and fleet policy.
- Keep semantic request/model state distinct from performance policy.
- Keep the fast request scheduler cheap. Expensive planning, profiling, and tuning stay off the per-step hot path.
- Prefer persistent request state, incremental metadata updates, and asynchronous host/device execution over rebuilding work each step.
- Optimize hardware through backend-specific implementations rather than a lowest-common-denominator backend.
- Do not require every model implementation or kernel to be Rust.
- Do not assume all inference state is KV.
- Keep inference state separate from model-residency/weight-placement concerns.
- Do not make GGUF, Python, CUDA, or another provider/backend technology mandatory in the engine-wide core merely because it is useful for one path.
- Do not build a general ML compiler, training runtime, or datacenter scheduler as part of the initial engine.
- Optimized execution variants must be correctness-qualified before automatic selection.
- Benchmark before claiming performance, simplicity, memory, latency, or overhead advantages.

## Verification

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

CUDA-feature compile checks and nonignored adapter tests run in CPU CI. Device
execution and numerical qualification require a compatible NVIDIA host. See
`benchmarks/runtime-contract.md`; do not confuse compilation with GPU evidence.
