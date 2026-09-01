# Engine

Engine is the temporary working name for a Rust-first model inference engine focused on low-overhead serving, state-aware execution, hardware-specialized backends, and safe live runtime policy.

The project is intended to stand alone. It must not require Archon or any future training runtime. External orchestrators may allocate resources and lifecycle-manage Engine deployments through generic interfaces, while Engine remains responsible for inference-local scheduling, state, execution planning, parallelism, and hardware execution.

## Current status

Early architecture and implementation bootstrap. The first target is a competitive single-GPU NVIDIA path with repeatable comparisons against current inference engines before distributed breadth is added.

## Build and verify

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Project context

Durable architecture, research, and roadmap context is centralized in the private [`nijaru/agent-context`](https://github.com/nijaru/agent-context/tree/main/projects/github.com/nijaru/engine/ai) repository rather than duplicated here.

Start with:

- [`brief.md`](https://github.com/nijaru/agent-context/blob/main/projects/github.com/nijaru/engine/ai/brief.md)
- [`spec.md`](https://github.com/nijaru/agent-context/blob/main/projects/github.com/nijaru/engine/ai/spec.md)
- [`PLAN.md`](https://github.com/nijaru/agent-context/blob/main/projects/github.com/nijaru/engine/ai/PLAN.md)
- [`DECISIONS.md`](https://github.com/nijaru/agent-context/blob/main/projects/github.com/nijaru/engine/ai/DECISIONS.md)

The eventual public/project name is intentionally unresolved; `engine` is only the working repository name.

## License

Apache-2.0.
