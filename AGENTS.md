# Engine

Rust-first model inference runtime and serving engine. `engine` is a temporary working name.

## Persistent context

Durable project context is centralized at:

`~/github/nijaru/agent-context/projects/github.com/nijaru/engine/ai/`

Do not recreate a repository-local `ai/` tree or duplicate product/design/roadmap prose here.

## Current repository workflow

During current private R&D, work directly on `main`. Do not open pull requests unless the user explicitly asks for one.

## Session start

1. Read centralized `brief.md`.
2. Read `STATUS.md` before making implementation claims.
3. For the current local implementation continuation, read `HANDOFF.md`.
4. Read the canonical context file relevant to the task.
5. Check `git status` before editing.
6. Keep implementation claims benchmark-backed.

## Context load map

| Task | Read |
|---|---|
| Product/architecture/scope | `design/runtime-scope.md`, then `spec.md` |
| Current local-session continuation | `HANDOFF.md` |
| Current implementation | `STATUS.md`, then code/tests |
| Current implementation sequence | `PLAN.md` |
| Durable choices | `DECISIONS.md` |
| Origin/product/business strategy | `research/origin-product-and-market-2026-09-01.md` |
| Benchmark methodology/Qwen3.8 target | `research/benchmark-tooling-2026-09-01.md`, then repository `benchmarks/` |
| Archon/external orchestration boundary | `design/orchestrator-boundary.md` |
| Ecosystem/training research | `research/ecosystem-and-adjacent-runtimes-2026-09-01.md` |

## Current first target

- Qwen3.8-27B, text language path first.
- RTX 4090 is the first performance machine.
- The user already has an Unsloth Q4 Qwen3.8-27B GGUF running with llama.cpp; inspect and pin that exact local artifact/configuration before selecting another quant or recording benchmarks.
- Qwen3.8 requires hybrid recurrent/linear-attention state plus full-attention KV state. Do not design a KV-only StateManager.
- GGUF is an initial local artifact, not a permanent core Engine representation.

## Architecture guardrails

- Engine owns the inference-engine/runtime layer itself: model loading/execution, quantized/local/offline inference, serving, scheduling/batching, model-local state/cache, backends/kernels, profiling/runtime policy, and eventually distributed inference.
- Engine must remain independently deployable under bare metal, containers, Kubernetes, Slurm, Archon, or another orchestrator.
- External orchestrators own physical resource allocation, machine placement, fleet health, and datacenter-wide resource policy.
- Do not import Archon implementation types into Engine core.
- Keep semantic model/request state distinct from performance policy.
- Keep the fast request scheduler cheap; planning/autotuning must not become a global optimizer in the token/work hot path.
- Optimize hardware through backend-specific implementations rather than a lowest-common-denominator core.
- Do not require every model implementation or every kernel to be Rust.
- Do not assume all inference state is KV.
- Do not make GGUF, Python, Mojo/MAX, or another framework a mandatory engine-wide dependency merely because it is useful for one provider/backend.
- Do not build a general ML compiler, training runtime, or datacenter scheduler as part of the initial engine.
- Preserve cheap low-level compatibility with a possible future training runtime, but do not distort inference abstractions around hypothetical training requirements.
- Benchmark before claiming performance, simplicity, or overhead advantages.

## Initial verification

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
