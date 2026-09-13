# Architecture

Ribn is a Rust-first general model inference runtime and serving engine. The
current implementation is an experimental autoregressive Qwen/CUDA path; it is a
qualification vehicle, not the architectural scope of the project.

[Inference engine design](inference-engine-design.md) is the target architecture.
[Roadmap](roadmap.md) sequences the migration and proof gates. Historical runtime
decisions remain in [runtime redesign](runtime-redesign.md).

## Current versus target architecture

Today:

```text
CLI
 |
ribn-text
 raw/chat/token preprocessing
 |
ribn Engine
 token-generation scheduling
 |
GenerationExecutor
 |
Qwen execution + NVIDIA
```

The low-level `ribn::Engine` is specifically an **autoregressive token runtime**.
Its contracts contain encoded token requests, Prefill/Decode work, committed token
prefixes and generated token events. Those are valid concepts for AR generation,
but they are not universal inference concepts.

The target is:

```text
CLI / Rust / Python / HTTP
            |
       loaded Model
 processor + capabilities + model pipeline
            |
        Orchestrator
 request/cancel/routing/output ordering
            |
  specialized stage runtimes
  AR | encoder/pooling | iterative/diffusion | ...
            |
 model/backend resources + execution
            |
 kernels / devices / collectives
```

A single-stage model takes a direct path through one runtime. Logical stages do
not imply processes or serialization. Multi-stage models can compose encoders,
AR decoders, diffusion/denoising, media decoders or other real execution units
without teaching one scheduler every possible model mechanism.

## Package status

| Package | Current responsibility | Direction |
| --- | --- | --- |
| `ribn` (`crates/runtime`) | AR request lifecycle, scheduling, bounded output, cancellation, executor contract | Preserve/evolve as the AR runtime; it should no longer define all of Ribn |
| `ribn-text` (`crates/text`) | Raw/chat/token input, tokenizer/template handling, incremental text decode, synchronous generation facade | Keep useful text behavior but remove Qwen/GGUF/CUDA loading assumptions; public model facade moves above model selection |
| `engine-qwen` | Qwen config, GGUF mapping, current CUDA executor | Remains a model integration/qualification path |
| `engine-gguf` | GGUF metadata/tensors and current tokenizer support | One artifact format; add HF config/safetensors/tokenizer/processor paths rather than making GGUF universal |
| `engine-nvidia` | NVIDIA resources, state and kernels | Backend-specific execution; continue CUDA Rust migration and optimization |
| `engine-core` | Legacy execution/state/runtime contracts | Transitional; retire after new boundaries replace its remaining uses |
| `ribn-cli` | Current `inspect`, experimental `run`, legacy `local` oracle | Grow into familiar local/serve/bench workflows over the general model facade |

Production dependency direction is checked by `tools/check-boundaries.py`. The
checker protects against accidental coupling; it should evolve with the target
architecture rather than freezing the current crate graph.

## Public model boundary

Applications should load a model and invoke supported operations. They should not
select an internal runtime family or construct scheduler batches.

The intended high-level Rust object is a concurrent/cloneable loaded `Model`
handle (exact naming remains open). Operation-specific methods express semantics:
raw generation/chat, embeddings/scoring, transcription, speech/media generation,
and similar APIs as support lands. A model can report its capabilities for
inspection and server route enablement.

The current `TextModel` is a temporary first facade. It hardcodes Qwen GGUF on
CUDA and borrows itself mutably while streaming; both are implementation gaps, not
public-design commitments.

HTTP follows established protocols where applicable. Chat/Completions/Responses,
embeddings/reranking, transcription/speech, and image/video APIs have different
semantics. Internal stage or capability selection must not become a user-facing
"task defaults" workflow.

## Model integration and artifact loading

Model architecture and artifact format remain separate concerns. The current
`QwenConfig` / `QwenGguf` split is worth preserving.

A model integration may own configuration validation, parameter mappings, input
processor behavior, logical pipeline topology, model-specific state semantics,
supported operations, and backend implementations. Shared serving, request
lifecycle and protocol code should be reused.

Hugging Face model directories/repository IDs, config JSON, safetensors, tokenizer
and processor metadata need first-class support. GGUF remains important for local
quantized inference but is not the canonical engine representation.

Ribn should research an optional compatibility/reference model path for model
bring-up. Native optimized Rust remains the production-performance target; a
reference path must not silently become a required Python serving dependency.

## Inputs, outputs and multimodality

Raw media is not a token scheduler concept. Public/request processing may accept
text/messages/token IDs, images, audio, video, precomputed embeddings or other
supported payloads. The model processor maps them to the concrete tensors,
embeddings, placeholder ranges, masks and metadata its stages require.

Outputs can be text/token streams, embeddings, scores/classes, audio, images,
video or model-specific structured results. Cross-stage payloads preserve request
identity and typed metadata while allowing co-located stages to keep large tensors
on device.

Media URL fetching/security belongs to the server layer. Processor/encoder result
caching should key on content and processing/model identity, not merely on a URL.

## Orchestrator ownership

The general orchestrator owns cross-stage lifecycle only:

- request/model identity;
- stage routing and correlation;
- cancellation/deadline/failure propagation;
- stage readiness and later replica affinity;
- bounded cross-stage queues/backpressure;
- output ordering and terminal completion.

It does not choose AR token batches, diffusion timesteps, KV pages or MoE kernels.
Stage runtimes own their execution-specific policy. Keep this control plane shallow;
a single-stage request should not traverse redundant Worker/Executor/Engine layers.

## AR runtime invariants

The existing ownership work remains useful inside the AR runtime.

`RequestId` is user-visible request identity; `SequenceId` identifies executor-owned
continuation state. Submission is not completion. Completion is validated before
logical prefix/output mutation. Cancellation never proves device completion or
permits live resources to be freed. Failed cleanup retains a retry owner; uncertain
teardown prefers retaining resources to unsafe destruction.

These are correctness invariants independent of scheduler policy.

The current scheduler's separate prefill/decode queues, one in-flight batch and
`Admission::Ready/Deferred` contract are **not** invariants. They are prototype
policy/mechanics to revisit as dynamic resources, cache reuse, speculation and
async overlap land. A unified scheduled-token budget is a strong candidate for the
future AR scheduler.

## AR continuation resources

Do not assume continuation state is KV. Full attention, sliding-window attention,
MLA, sparse attention, recurrent/SSM state, draft state and other model mechanisms
may coexist.

Dynamic resource management needs to cooperate with scheduling around allocation,
prefix match/reuse, per-step preparation/update, eviction, preemption and release.
Physical pages/checkpoints/tensors stay model/backend-owned; scheduling sees
capacity and cost.

A prefix hit is valid only when all continuation components required at that
semantic boundary are valid. For Qwen's current hybrid state, matching KV without
matching recurrent state is not a reusable continuation.

The existing full-context-per-sequence reservation in `engine-qwen` is an explicit
prototype limitation.

## Host/device execution

The target serving path is async-first and data-oriented:

- one mutable runtime owner with bounded command submission from cloneable handles;
- stable request rows/slots for active lifetimes;
- persistent host/device metadata with incremental updates;
- packed/ragged batches and reusable workspace;
- no normal-path host/device synchronization;
- explicit completion events and CPU/GPU scheduler overlap;
- device-side sampling/output processing where useful;
- qualified CUDA graph/kernel variants with correct fallbacks;
- backend-specific layouts and kernels rather than a lowest-common-denominator
  device abstraction.

Coarse trait-object dispatch between a prepared stage/runtime and the orchestrator
is not a performance concern by itself. Avoid dynamic dispatch and allocation in
per-layer/per-token hot kernels unless measurement justifies it.

## Hardware and distribution

The RTX 4090 is development/qualification hardware only. NVIDIA CUDA is the first
backend, not an engine-wide requirement. AMD, Metal/Apple and CPU backends should
be able to specialize their execution rather than imitate CUDA abstractions.

Ribn may own inference-local distributed execution: tensor/expert/pipeline/data or
context parallelism, stage-local replication, prefill/decode disaggregation and
model-state transport. External orchestrators allocate/place resources. Logical
model topology stays separate from deployment topology.

## Current limitations

Ribn currently has only the Qwen GGUF/CUDA AR path in production use. `TextModel::load`
rejects non-Qwen GGUFs; the AR executor accepts only token-generation requests; state
is reserved for full configured context per active sequence; the scheduler has one
batch in flight; sampling support is limited.

Provisional pressure-test scaffolding exists beyond that path for format-level
SafeTensors validation, local HF-style package resolution, and a non-AR batch runtime
with a BERT reference path, but none of it is production model support. There is no
general model/architecture registry, production processor or multimodal prepared-input
path, encoder/pooling runtime, diffusion runtime, cross-stage orchestrator, concurrent
public model handle, HTTP server, second hardware backend or distributed execution.

`docs/execution-foundation.md` records what has actually been validated and is the
single place to update when that changes. [Resource, submission and snapshot
protocol](resource-protocol.md) records the contracts the runtimes above that
foundation need next. This section states the shape of the gap rather than restating
evidence, so the two cannot drift.

Those limitations are reasons to redesign now while the codebase is small, not
reasons to expand Qwen-specific abstractions until they become harder to remove.
