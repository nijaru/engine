# Ribn agent guidance

Ribn (pronounced “ribbon”) is a Rust-first **general model inference engine**. The
codebase is evolving quickly; current repository code, tests, and docs are more
authoritative than older handoffs or remembered architecture.

## Repository workflow

Work directly on `main` unless the maintainer asks for a branch or pull request.
Keep commits focused and leave the tree green. Public architecture/roadmap decisions
belong in `docs/`; do not make repository behavior depend on private context files.

At session start, read `README.md`, `docs/inference-engine-design.md`,
`docs/architecture.md`, and `docs/roadmap.md`, then inspect the actual code/tests
relevant to the change. Keep correctness and performance claims tied to evidence.

## Product direction

The goal is a state-of-the-art inference engine in Rust, not a Qwen runner or a
novel interaction model. The current Qwen3.8 GGUF/CUDA path and RTX 4090 are a hard
qualification workload, not the product boundary.

Ribn should progressively support the major current inference classes: decoder-only
and hybrid AR models, multimodal/omni models, encoder/pooling/reranking models,
encoder-decoder speech models, diffusion/image/video generation, and other execution
regimes demonstrated by real models. Supporting a class means implementing and
qualifying it; do not claim arbitrary checkpoint compatibility from architecture
alone.

Prefer familiar local, library, Python, and serving workflows unless Ribn has a
concrete reason to differ. Do not add a user-visible task-default ceremony merely
because internal runtimes differ. Applications should load a model and call the
operation they intend: generate/chat, embed/score, transcribe, synthesize, generate
media, etc. Useful low-level interfaces remain valid for advanced embedding.

## Architecture direction

The current `crates/runtime` (`ribn`) is an **autoregressive token runtime**. Preserve
its useful ownership/cancellation work, but do not treat `GenerationExecutor`,
`TokenRequest`, `Prefill/Decode`, or token events as universal inference contracts.
Another runtime redesign is appropriate when pressure tests show these are the
wrong top-level substrate.

Target decomposition:

- a loaded-model/model-package boundary resolves architecture, artifact, processor,
  supported operations, logical stage topology, model state semantics, and backend
  variants;
- a shallow orchestrator owns request identity, cross-stage routing, cancellation,
  bounded queues, failure propagation, and output ordering;
- specialized runtimes own execution-specific scheduling (AR token generation,
  batch/encoder/pooling, iterative diffusion/media, or additional regimes justified
  by actual models);
- backend/model code owns physical resources, kernels, layouts and collectives;
- CLI/HTTP/Python/Rust frontends reuse model/application behavior rather than
  reimplementing model execution.

Logical stages are not mandatory processes. Single-stage models should take a
short direct path. Avoid pass-through Worker/Executor/Engine layer stacks.

## Model and artifact support

Model configuration and artifact representation are separate concerns. The current
`QwenConfig` / `QwenGguf` split is directionally correct.

Hugging Face repository IDs/local directories, config JSON, safetensors, tokenizer
and processor metadata should become first-class. GGUF remains important for
quantized/local use but is one loader path, not a universal representation.

Research a compatibility/reference backend for rapid model bring-up and independent
correctness comparison. Native optimized execution is the production-performance
target; do not introduce a mandatory Python/PyTorch serving dependency solely for
fallback coverage.

Typed public inputs may include text/messages/token IDs, images, audio, video,
embeddings/tensors where supported, and future media/action payloads. Model
processors map these to concrete tensors/placeholders/positions. Do not model raw
media as fictional text tokens in the common API. Outputs may likewise be text,
tokens, embeddings, scores, audio, images, video, latents, or structured results.

## AR runtime, performance and state

Within the AR runtime, keep request/device hot paths cheap. Prefer persistent
request/device rows, incremental metadata updates, asynchronous host/device
execution, packed/ragged batches, reusable workspaces, and backend-specific optimized
paths where measurements support them.

Do not assume continuation state is KV. Full/SWA/MLA/sparse attention, recurrent or
SSM state, speculative/draft state and other components can coexist. Future dynamic
resource management must cooperate with scheduling around allocation, prefix reuse,
per-step preparation/update, eviction, preemption and release while physical
layouts remain model/backend-owned.

A semantic prefix is reusable only when all required continuation components agree
at that boundary. Qwen's current hybrid state is a useful test of this invariant.

The current separate prefill/decode queues, one in-flight batch, static
`Ready/Deferred` admission, and full-context-per-sequence reservation are prototype
mechanics, not architectural commitments. Revisit them once the real resource model
exists; unified token-budget scheduling is one strong candidate.

Optimized variants need correctness qualification for the scope in which they are
automatically selected. Compilation or host tests are not GPU evidence.

## Hardware and distribution

NVIDIA single-GPU execution is first; the development GPU is qualification hardware,
not an engine-wide constraint. AMD, Apple/Metal, CPU and future backends should be
allowed to specialize rather than imitate CUDA abstractions.

Ribn may own inference-local tensor/expert/pipeline/data/context parallelism,
collectives, prefill/decode disaggregation, stage replication and model-state
transport. External orchestrators own resource allocation and fleet placement.
Logical model topology must remain distinct from deployment topology.

## Pressure tests

Before treating new top-level contracts as stable, exercise them with materially
different paths: current Qwen hybrid AR, a dense decoder-only model, an encoder or
embedding model, an encoder-decoder speech model, a VLM, a diffusion image/video
model, a non-AR text model if practical, and eventually another hardware backend.
Small reference-backed implementations are enough to expose a wrong boundary; full
optimized support is not required for every pressure test.

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
