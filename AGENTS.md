# Ribn agent guidance

Ribn (pronounced “ribbon”) is a Rust-first **general model inference engine**. The
codebase is evolving quickly; current repository code, tests, and docs are more
authoritative than older handoffs or remembered architecture.

## Repository workflow

Work directly on `main` unless the maintainer asks for a branch or pull request.
Keep commits focused and leave the tree green. Public architecture/roadmap decisions
belong in `docs/`; do not make repository behavior depend on private context files.

At session start, read `README.md`, `docs/inference-engine-design.md`,
`docs/execution-foundation.md`, `docs/pipeline-composition.md`,
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
  supported operations, logical topology, model state semantics, and backend
  variants;
- a shallow orchestrator owns genuine cross-runtime dependencies, cancellation,
  bounded queues, failure propagation, placement and output ordering;
- specialized runtimes own execution-specific scheduling (AR token generation,
  batch/encoder/pooling, iterative diffusion/media, session/realtime, or additional
  regimes justified by actual models);
- backend/model code owns physical resources, kernels, layouts and collectives;
- CLI/HTTP/Python/Rust frontends reuse model/application behavior rather than
  reimplementing model execution.

Logical stages are not mandatory processes. Single-stage models should take a
short direct path. Avoid pass-through Worker/Executor/Engine layer stacks.

Do not force every encoder behind a top-level stage boundary. Sequential
encoder-decoder models can use a staged handoff; VLM/omni models may need a coupled
runtime where encoder readiness/cache and AR prompt progress participate in the same
resource decision. See `docs/pipeline-composition.md`.

Architecture extensibility is not a purity constraint. A central Rust enum,
registry, factory, capability record, or shared-code change is acceptable when it is
the simplest representation of actual supported architectures. Avoid scattered
model-family branches through unrelated scheduling/server code; do not invent a
plugin ABI or generic framework merely so adding a model never changes central code.

## Shared execution foundation

`crates/foundation` (`ribn-foundation`) is a provisional pressure-test of the
shared layer beneath inference policy. It currently models parameter identity and
versions, physical materializations, resource topology, and prepared placement.
Its types are **not stable APIs** and should change when real model/backend work
shows a better representation.

Keep this boundary neutral where reuse is natural:

- inference-specific concepts such as requests, tokens, KV caches, prefix reuse,
  serving, or continuous batching must not become requirements for parameter,
  device, operator, collective, placement, or storage primitives;
- future training compatibility does not justify adding gradients, `requires_grad`,
  autograd tape, optimizer state, losses, backward graphs, or training schedulers
  to Ribn inference;
- logical parameter identity is distinct from checkpoint representation, dtype,
  quantization, sharding, layout, device placement, and storage materialization;
- inference work and reusable state must be associated with a coherent parameter
  version rather than assuming weights are immutable forever;
- logical model topology is distinct from deployment topology so the same model
  semantics can be prepared for one device or many devices/nodes.

Do not turn the transitional `engine-core` into this foundation by expanding its
prototype enums. New shared primitives should be justified independently, and old
core contracts can be removed as replacement paths qualify.

`crates/batch` (`ribn-batch`) is likewise a design-validation runtime, not a final
embedding scheduler. It proves non-AR work need not inherit token/prefix/KV
semantics. A variable-length reference encoder showed that request count alone
cannot safely form all batches, so the concrete executor can shorten the oldest
FIFO candidate set using its own shape/memory/compute constraints. The actual BERT
architecture pressure test then ran embeddings, self-attention, residual/LayerNorm,
FFN and pooler semantics through the same boundary without requiring a new universal
cost abstraction. Do not replace executor-owned constraints with a universal work
unit unless real device workloads justify it. Reordering, bucketing,
heterogeneous batching, async execution and resource-aware admission remain open.

A future trainer may reuse lower-level parameter/device/operator/collective
infrastructure, but it remains a separate execution system. Shared infrastructure
and shared physical representations are not the same requirement.

A semantic-operator experiment currently exists only as a test: semantic matching
chooses specialized versus reference RMSNorm during preparation, after which normal
execution does not repeat support-predicate/registry lookup. Do not promote that
fixture into a production IR/operator framework until real model/backend work shows
that it reduces duplication without adding hot-path dispatch or compiler machinery.

## Model and artifact support

Model configuration and artifact representation are separate concerns. The current
`QwenConfig` / `QwenGguf` split is directionally correct.

`crates/safetensors` provides a thin format adapter that validates artifacts and
exposes format-level tensor views without assigning model semantics or allocating
execution tensors. `crates/hf` resolves local HF-style `config.json` plus single or
sharded SafeTensors weights without choosing architecture/runtime/backend. Preserve
that separation as remote repository/revision, tokenizer and processor support are
added.

The BERT architecture pressure test is now concrete evidence for this separation:
model code interprets the raw HF config and parameter names, validates expected
shapes, and executes the architecture while the package/artifact layers remain
model-agnostic. It also exposed a practical loader requirement: a resolved
SafeTensors shard should be opened/owned once and serve repeated tensor views rather
than rereading the whole shard for every parameter. Promote that reusable ownership
mechanism without moving BERT/Qwen semantics down into `ribn-hf`.

Hugging Face repository IDs/local directories, config JSON, SafeTensors, tokenizer
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

`RequestId` and `SequenceId` are deliberately different identities. `RequestId` is
the stable AR runtime request identity; `SequenceId` identifies executor
continuation ownership. `GenerationExecutor::admit` receives both so model-owned
prepared request state or tracing can correlate with the logical request without
making `TokenRequest` a generic payload container. Do not put raw media or a vague
universal multimodal object into `TokenRequest` merely because this seam exists.

Sequential cross-runtime tests establish the current ownership rule: before AR
admission, prepared state belongs to the producer/orchestrator; after successful
admission, it belongs to executor sequence state and is reclaimed through the normal
`release` path. Do not add a generic prepared-state cleanup trait unless another
real integration demonstrates that this ownership split is insufficient.

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

The VLM prompt-position pressure test demonstrates that coupled generation can need
scheduler-visible model-prepared feature identity/span/readiness plus **separate**
encoder-compute and encoder-cache constraints. Those integer test budgets are not a
proposal for an arbitrary generic resource vector. Let an actual VLM integration
determine the minimum production scheduler/resource-planner seam. Raw media and
processor implementation details remain above it.

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

Current validation evidence includes:

- the same logical model can be placed against local or multi-node resource
  topologies;
- non-AR work can execute without token/KV semantics while remaining pinned to a
  parameter version;
- variable-length encoder inputs forced executor-informed FIFO batch selection;
- SafeTensors + local HF-style package code keeps artifact semantics separate from
  model semantics;
- an actual BERT architecture reference path executes over that package boundary and
  confirms sequence-length batching without another common scheduling abstraction;
- sequential batch-encoder -> AR handoff passes prepared state in-process, correlates
  by AR `RequestId` independent of handoff order, and transfers cancellation cleanup
  ownership at admission;
- VLM prompt-position tests show encoder dependencies can interleave with AR prefill
  and that encoder compute and cache capacity are distinct scheduling concerns.

Next evidence-driven pressure points are masked/padded/ragged encoder/device
execution, ordinary architecture resolution using the concrete Qwen+BERT cases, a
genuine encoder-decoder model, and an actual VLM/processor integration. Prefer those
over another synthetic framework layer. The actual VLM should determine whether the
AR scheduler needs explicit per-item dependency descriptors, a model/resource
planner, or another representation.

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
