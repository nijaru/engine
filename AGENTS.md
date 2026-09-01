# Engine

Rust-first model inference engine. `engine` is a temporary working name.

## Persistent context

Durable project context is centralized at:

`~/github/nijaru/agent-context/projects/github.com/nijaru/engine/ai/`

Do not recreate a repository-local `ai/` tree or duplicate product/design/roadmap prose here.

## Current repository workflow

During current private R&D, work directly on `main`. Do not open pull requests unless the user explicitly asks for one.

## Session start

1. Read centralized `brief.md`.
2. Read `STATUS.md` before making implementation claims.
3. Read the canonical context file relevant to the task.
4. Check `git status` before editing.
5. Keep implementation claims benchmark-backed.

## Context load map

| Task | Read |
|---|---|
| Product/architecture/scope | `spec.md` |
| Current implementation | `STATUS.md`, then code/tests |
| Current implementation sequence | `PLAN.md` |
| Durable choices | `DECISIONS.md` |
| Origin/product/business strategy | `research/origin-product-and-market-2026-09-01.md` |
| Benchmark methodology/tooling | `research/benchmark-tooling-2026-09-01.md`, then repository `benchmarks/` |
| Archon/external orchestration boundary | `design/orchestrator-boundary.md` |
| Ecosystem/training research | `research/ecosystem-and-adjacent-runtimes-2026-09-01.md` |

## Architecture guardrails

- Engine must remain independently deployable under bare metal, containers, Kubernetes, Slurm, Archon, or another orchestrator.
- Engine owns inference-local request scheduling, batching, model state, execution planning, parallelism, speculation, execution variants, and backend execution.
- External orchestrators own physical resource allocation, machine placement, fleet health, and datacenter-wide resource policy.
- Do not import Archon implementation types into Engine core.
- Keep semantic model/request state distinct from performance policy.
- Keep the fast request scheduler cheap; planning/autotuning must not become a global optimizer in the token hot path.
- Optimize hardware through backend-specific implementations rather than a lowest-common-denominator core.
- Do not require every model implementation or every kernel to be Rust.
- Do not build a general ML compiler, distributed KV platform, training runtime, or datacenter scheduler as part of the initial engine.
- Preserve cheap low-level compatibility with a possible future training runtime, but do not distort inference abstractions around hypothetical training requirements.
- Benchmark before claiming performance, simplicity, or overhead advantages.

## Initial verification

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
