# Inference engine design

Reviewed: 2026-09-13 (America/Los_Angeles)
Status: authoritative target architecture. Boundaries below are accepted direction;
exact public signatures and preparation/wakeup types remain open until their roadmap
gates pass. This is not a claim of implemented support.

## Reference ownership and implementation rules

Read these three references for architectural work, in order:

1. This document owns product priorities, decomposition and public API semantics.
2. [Resource protocol](resource-protocol.md) owns execution/resource ownership and
   state-transition rules. Do not reinterpret them in a frontend or model adapter.
3. [Roadmap](roadmap.md) owns sequencing, unresolved decisions and acceptance gates.

[Architecture](architecture.md) describes current code, not competing target rules.
[Execution foundation](execution-foundation.md) and
[pipeline composition](pipeline-composition.md) retain experiments and limitations;
their provisional types are not required implementation architecture.
[Research agenda](research-agenda.md) lists investigation topics, not an alternate
backlog. Historical designs do not override these references.

Before implementing an architectural slice, resolve its externally observable
contract: ownership, permitted transitions, failure scope, cancellation, boundedness,
readiness and shutdown. Record any unresolved choice explicitly in the roadmap.
Choose internal data structures, helper names and qualified kernel details locally.
If source or a counterexample contradicts a contract, revise the owning reference
and its tests rather than silently weakening the contract or layering a workaround.

Design enough to implement the next vertical slice safely, not every future trait.
A small executable counterexample may precede final API signatures; label it as an
experiment and do not build production dependents on unresolved contracts. A gate
closes only with the stated evidence, not with compilation or more documentation.

## Decision

Ribn is a **server-first**, general model inference engine. Optimize and qualify
sustained concurrent execution, throughput within latency objectives, bounded memory,
fairness, overload and cancellation before local startup convenience. In-process
Rust/Python and local CLI use share the same engine; server-first does not require
HTTP, IPC or a multiprocess single-device deployment.

All v0 APIs are unstable. Replace flawed contracts and remove obsolete paths rather
than preserving them with compatibility shims. Preserve correctness evidence, not
historical architecture. Future training and distribution should require localized
extensions; this is a goal to pressure-test, not a promise of zero future refactors.

The current token-generation runtime is
an **autoregressive (AR) runtime**, not the definition of the whole engine.

That distinction matters because current inference workloads include decoder-only
language models, hybrid attention/SSM models, encoder-only embedding/reranking
models, encoder-decoder speech models, multimodal encoders feeding AR decoders,
omni models with several model components, diffusion/image/video generation, and
non-AR text generation. One token scheduler cannot efficiently or cleanly express
all of those execution loops.

The target is therefore:

```text
CLI / Rust / Python / HTTP
            |
     application APIs
  familiar operation-specific methods
            |
      loaded Model handle
 processor + capabilities + pipeline
            |
        Orchestrator
 request identity / cancellation / routing
 output ordering / cross-stage lifecycle
            |
   +--------+---------+----------------+
   |                  |                |
 AR runtime       batch/encoder    iterative runtime
 token progress    and pooling     diffusion / other
 continuation      execution       repeated-step models
 scheduling
   |                  |                |
   +----------- model/backend execution -----------+
                      |
       resources / kernels / collectives / devices
```

A model may have one stage. The common case must not pay for IPC, serialization,
or graph traversal merely because multi-stage models exist. Stages are logical
execution units, not mandatory processes. A model package can compose several
stage runtimes when its real architecture requires them.

This is a deliberate redesign of the top-level boundary, not a rejection of the
existing AR work. The request/sequence ownership, cancellation, bounded output,
and conservative device-lifetime behavior already implemented are useful inside
the AR runtime.

## Scope and success criterion

"General model inference" does **not** mean any arbitrary checkpoint runs without
implementation work. That is not realistic for a native optimized Rust engine.
It means that the engine architecture can support the major current inference
classes without another foundational rewrite, and that adding a model normally
changes model/processor/backend code rather than duplicating serving infrastructure.

A strong model-support strategy has two layers:

1. native optimized implementations for production performance;
2. a separately evaluated compatibility/reference path for faster bring-up and
   correctness comparison when practical.

vLLM can use compatible Hugging Face Transformers implementations as a fallback.
Rust has no equally broad native equivalent today, so Ribn should research a
reference/compatibility backend rather than pretending a registry alone provides
day-zero model support. Python/PyTorch must not become a mandatory production
runtime merely to obtain this fallback.

## Public API and UX

Users should interact with model capabilities, not internal runtime categories.
There is no public "task default" ceremony.

Expected surfaces:

| Surface | Direction |
| --- | --- |
| Local CLI | `ribn run <model>`, plus inspection/benchmark/device utilities and operation-specific commands only when their semantics genuinely differ |
| Server | `ribn serve <model>` with tested OpenAI/Anthropic-compatible endpoints where applicable, health/readiness, metrics, cancellation, bounded admission, and documented extensions |
| Rust | Cloneable loaded-model handle; async generation/streaming and batch APIs; operation-specific methods such as generate/chat/embed/score/transcribe rather than a mandatory generic task selector |
| Python | In-process bindings over the same engine for evaluation, RL/post-training, offline inference, and integration with the Python ML ecosystem |
| Low level | Optional direct runtime/model interfaces for embedding, custom schedulers, token-level control, profiling, or specialized applications |

Serving endpoints should follow the operation they represent. Chat/Responses are
appropriate for conversational and heterogeneous generative requests; embeddings,
reranking, transcription, speech synthesis, image generation, and video generation
have different input/output semantics and should use their expected APIs. Protocol
adapters do not define internal runtime types.

Configuration should distinguish model/load settings, execution/resource settings,
scheduling policy, request-generation settings, and server settings. Defaults are
convenience, not hidden task selection. Explicit request values override resolved
model defaults where supported.

## Greenfield review: inference first, training-compatible below policy

The from-scratch design would have four boundaries, not a universal engine graph:

1. **Artifact and model definition:** validated configuration, processors, parameter
   mappings and operation semantics; no serving lifecycle.
2. **Prepared execution:** an owning backend-specific model snapshot, device storage,
   qualified kernels and explicit completion/resource ownership.
3. **Execution policy:** AR, batch and later iterative/training loops, each with its
   own scheduling and state semantics. Share mechanisms only when implementations
   demonstrate reuse.
4. **Application access:** operation-specific owned requests/results, bounded
   submission, cancellation, diagnostics and frontend adapters.

A single-model request should cross the application boundary once, not pass through
an orchestrator, worker, executor and engine that forward identical calls. Genuine
multi-runtime dependencies justify orchestration; a model with one runtime does not.

### Rust API decision

Keep two deliberate surfaces. The application surface is an owned cloneable handle
backed by a single mutable execution owner. The direct surface lets an embedding
application drive that same runtime without channels or a background thread.
Do not implement synchronous and asynchronous execution as separate schedulers.

The intended usage is `model.generate(request).await?` or
`model.stream(request).await?`, returning an owned response or request stream that
can outlive the borrow used to submit it. These are target signatures, not working
examples. A stream owns cancellation interest, yields ordered deltas and one terminal
outcome, and supports explicit cancellation. Dropping it abandons delivery and asks
the execution owner to retire the request; it never waits for GPU completion.

Use a model loader for source/revision/device preparation and operation-specific
handles or checked methods for supported capabilities. Do not expose a giant
`infer(Any) -> Any` API, nor implement unsupported operations as silent fallbacks.
Introduce typed media and tensor interchange when a real operation needs them.
Borrowed input is convenient for direct calls; queued input must be owned or share
an explicitly immutable allocation. Keep backend tensor types out of network APIs.

Separate load/resource settings, scheduler policy and per-request sampling settings.
Use ordinary structs/builders with validated defaults; avoid typestate builders for
unrelated configuration choices. Use typed errors distinguishing invalid input,
unsupported capability, overload, request failure and owner/device failure. Preserve
underlying error sources; strings alone are poor diagnostics and control flow.

Batch convenience must admit a bounded window and preserve input ordering; collecting
an arbitrary iterator before admission is not bounded batching. Offer incremental
results for large offline workloads. A stalled stream must backpressure its request
without blocking cancellation or unrelated consumers. Preprocessing also needs byte
limits and bounded concurrency; moving tokenization before enqueue does not bound it.

A worker thread is a practical first execution owner for thread-affine/blocking GPU
APIs. Async clients await bounded channels; they do not block an async executor with
GPU polling or tokenization. Keep a blocking wrapper for CLI/local use and make its
runtime restrictions explicit. The core scheduler need not depend on Tokio. Adopt
one established channel/wakeup implementation instead of inventing synchronization.

### Proposed text cutover semantics (slice-2 audit, 2026-09-14)

These refinements are proposed for approval before implementation; the current
borrowed facade does not implement them. The [current-code audit](architecture.md#slice-2-boundary-audit-2026-09-14-84a146a)
records the conflicting paths. The token driver's accepted lifecycle remains owned
by the resource protocol.

- The text layer owns preprocessing and incremental decoding, not execution progress.
  CUDA loading assembles that layer with the existing owned token driver. A small
  processor seam must exercise the actual facade on the host, not another simulated
  lifecycle. Keep model-specific formatting outside the token runtime.
- Incremental offline batching yields ordered per-input results from a bounded window.
  Preparation/decode failure settles only that input; previously admitted peers remain
  owned and may complete. This intentionally replaces all-input preparation atomicity.
  A slow first item may hold later completed results, but those results retain window
  capacity; do not admit replacements merely because execution finished. Dropping the
  batch abandons its live streams without waiting for device completion.
- Ordinary token terminal events, including explicit cancellation, flush incomplete
  UTF-8 once before the text terminal. A decode error abandons only its request and
  yields one typed error. Owner failure drains already-delivered token events under
  the driver contract, then yields one owner error without inventing successful usage
  or an ordinary terminal flush. A decoder error encountered first settles that text
  stream; it does not recover the failed execution owner.
- An admission charge precedes processor work and remains owned while that work can
  still access retained input, even if the submission future is dropped. Bounded
  preprocessing must not block an async executor or add an unbounded work queue.
  Raw/message storage, template expansion, scratch, encoded output, decoded deltas
  and terminal staging need separate byte accounting. Fuel and token count alone do
  not establish those bounds. Immutable loaded vocabulary and caller-collected results
  must be explicitly distinguished from buffered application storage.

Before coding, settle concrete processor limits/enforcement, preprocessing cancellation
and shutdown, shared-handle overload during a batch window refill, and the collect
helper's result-size policy in the resource protocol. No universal processor framework
or second execution worker is implied by these requirements.

### Performance, UX and contribution consequences

| Decision | Benefit | Cost and required evidence |
| --- | --- | --- |
| Single execution owner | Local mutation, simpler cancellation, no global hot-path mutex | Queue/wakeup overhead; compare direct and handle paths with a trivial executor and real GPU |
| Owned request stream | Concurrent callers and predictable disconnect cleanup | Bounded per-request state; prove saturation and drop races |
| Coarse model dispatch | Simple registration and backend specialization | Dispatch once per batch/operation; do not assert zero cost or force per-layer virtual calls |
| Backend-native execution | Packed layouts, vendor kernels, graphs and fusion remain available | More backend implementation work; qualify scope and retain references |
| Operation-specific API | Familiar semantics and useful errors | Some surface growth as actual operations land, preferable to untyped payloads |
| Versioned executable ownership | Safe cache/update boundaries and future rollout integration | Drain latency or explicit double-buffer memory, not free hot swapping |

First optimize algorithm and data movement: scalable prefill GEMM versus small-M
GEMV, tiled attention, dynamic hybrid continuation capacity, packed cross-request
work, persistent metadata and reusable buffers. Then measure queue overhead, CPU
allocation, scheduling gaps and synchronization. Rust alone does not make a GPU
engine fast. Asynchronous APIs alone do not overlap CPU/GPU work.

Report load time/peak host and device memory, TTFT, inter-token latency, throughput,
latency tails, cancellation latency and stalled-consumer memory under matched
workloads. Keep direct-path host overhead separate from model execution. One 4090
prompt sweep cannot establish SOTA performance or a universal scheduling policy.

Contributors should implement model semantics, processors and backend components,
register the architecture centrally, then run a common lifecycle/numerical harness.
Do not promise every model is a forward-function plugin: a new state mechanism may
need a shared contract change. The second decoder and real VLM remain the proof.

### Training decision and reconsideration trigger

Training is a plausible subsequent product, not an inference mode. It needs losses,
backward execution, saved activations/rematerialization, gradients, optimizers,
precision/scaling policy, collectives and checkpoint/restart semantics. A token
scheduler and quantized forward implementation cannot provide those by extension.

Share artifact I/O, logical parameter identity, useful device/storage primitives,
completion dependencies and collectives where concrete implementations align. Allow
shared model semantics or a small operation vocabulary later; do not force optimized
inference through an eager tensor/autograd abstraction to reserve that option.
Training and inference may use different model programs and materializations while
sharing tested mathematical components. Native Rust control also does not require
rewriting working CUDA/vendor kernels into Rust immediately.

The first integration target should be trainer-to-rollout snapshot publication and
in-process Python tensor/result interchange, with explicit lifetime and version
ownership. A future trainer should first prove a small forward/backward/update loop
against an independent reference, then checkpoint recovery and distributed progress.
That experiment—not metadata types—determines whether a shared model/operator layer
actually saves work. Reconsider the split if maintaining two model semantics becomes
a demonstrated correctness or contribution bottleneck.

### Gap and alignment decision

At `2abf382`, the crate graph has useful format separation but the Qwen adapter still
translates `ribn::BatchItem` through `engine-core::ExecutionBatch` and state managers.
`ribn-foundation` mostly describes identities/topology rather than owning executable
storage. `TextModel` hardcodes Qwen loading, borrows mutably for streaming, flattens
errors and duplicates request cleanup. Completion planning clones token vectors and
allocates temporary row plans; `Blocked` is immediately resubmitted. These are source
observations, not inferred benchmark bottlenecks.

Align by replacing incomplete mechanisms, not by adding more layers around them:

1. Remove completion-time blocking until preparation can own real waiting; retain
   strictly positive partial progress and allocation-free whole-batch validation.
2. Move cancel-and-discard into the runtime mailbox owner and remove frontend cleanup
   queues. Exercise errors and abandoned terminal mailboxes with host regressions.
3. Add the bounded owned handle over this owner, with real wakeups, typed errors,
   host-testable text behavior and one pinned executable lifetime. No general registry
   or hot-swap machinery is needed to establish that lifetime.
4. Use the real asynchronous encoder to establish preparation/reservations and
   dependencies. Then introduce dynamic hybrid AR resources and real model additions.
5. Replace legacy Qwen/core translation at the physical execution boundary, preserving
   the qualification oracle. Delete obsolete helpers as their last consumers move;
   do not wait for all future model classes before deleting dead paths.

Keep existing provisional foundation/batch tests as counterexamples, not architecture
requirements. Do not enlarge them into production frameworks without real consumers.
The [resource protocol](resource-protocol.md) owns precise lifecycle and preparation
rules; the [roadmap](roadmap.md) owns sequencing and completion status.

## Model package boundary

A model family needs more than a forward function. A prepared model package may
provide:

- architecture/config validation independent of artifact format;
- weight-name/shape mappings and supported quantization encodings;
- tokenizer and model-specific processor behavior;
- chat/input templates and special-token semantics where relevant;
- supported public operations/capabilities;
- logical stage topology when the model contains multiple execution regimes;
- model-specific continuation/resource semantics;
- backend execution implementations and supported hardware/precision variants;
- output parsing/materialization where the model emits structured or media data.

Architecture registration is an internal selection mechanism. The user normally
passes a Hugging Face ID, local model directory, or supported artifact and Ribn
loads the appropriate implementation.

The current `QwenConfig` / `QwenGguf` split is directionally correct: architecture
semantics should not live in GGUF parsing. But GGUF must become one artifact path,
not the center of model loading. Hugging Face config, safetensors, tokenizer data,
processor metadata, and model revisions should be first-class because that is how
most new models are published. GGUF remains important for quantized/local use.

## Input and output model

The engine must not encode every modality as fictional text tokens at its public
boundary.

Application-level inputs need typed content: text/messages/token IDs, images,
audio, video, already-computed embeddings/tensors where supported, and future media
or action payloads. Model processors convert those inputs into the concrete tensors,
placeholder tokens, positions, masks, or encoder work the model needs.

Likewise, outputs may be text/token deltas, embeddings, scores/classes, audio
chunks, images, videos, latents, or model-specific structured outputs. Cross-stage
transport should preserve request identity, completion/error state, modality and
shape/format metadata without exposing backend-specific tensor classes in network
protocols.

Remote media fetching belongs to serving and needs explicit size/domain/security
policy. Media preprocessing/encoder outputs should be cacheable by content +
processor identity when that produces a real performance benefit.

## Orchestrator

The top-level orchestrator owns only cross-stage lifecycle:

- stable request identity and model/version identity;
- input dispatch after model processing;
- routing through the model's declared logical stages;
- cancellation, deadlines and failure propagation;
- output ordering and terminal completion;
- stage readiness/liveness and, later, placement/replica affinity;
- bounded queues and backpressure between stages.

It does **not** batch attention tokens, schedule diffusion timesteps, allocate KV
pages, or choose MoE kernels. Those policies stay in the stage runtime that has the
information to optimize them.

Keep this layer shallow. SGLang's 2026 omni roadmap is explicitly collapsing a
Stage -> Worker -> Executor -> Engine stack to Stage -> Engine; Ribn should avoid
creating equivalent pass-through layers. A single-stage model should have a direct
fast path through one stage runtime.

## Specialized runtimes

### Autoregressive runtime

The existing `crates/runtime` contract is a useful starting point for this runtime,
but it is not finished. It currently assumes encoded token input, Prefill/Decode
work records, one in-flight scheduler batch, and static sequence admission.

The target AR runtime needs to handle:

- dense/full attention, sliding attention, MLA and sparse attention;
- MoE and expert parallel execution;
- recurrent/SSM/linear-attention state alongside KV-like state;
- encoder-conditioned or multimodal prompt embeddings;
- paged/dynamic continuation resources and prefix reuse;
- chunked prefill and mixed/ragged batches;
- speculative methods including draft/verify/rollback and model-native MTP;
- structured/constrained decoding, logprobs, penalties and device-side sampling;
- preemption/recompute and resource-aware admission;
- asynchronous scheduler/GPU overlap and more than one in-flight execution where
  measurements justify it.

The current separate prefill/decode queues are policy, not an architectural
invariant. vLLM V1's unified scheduled-token budget is a strong candidate once
Ribn's dynamic resources exist. Prefill/decode can remain execution distinctions
without forcing separate scheduler queues.

### Batch / encoder / pooling execution

Embedding, classification, reranking, reward, vision/audio encoders and similar
models often need memory-aware dynamic batching but not an AR continuation loop.
They should not be forced through `GenerationExecutor`, fake `SequenceId` prefixes,
or token-output events.

Encoder-decoder models can compose an encoder stage with an AR decoder when that
matches the model. The boundary should allow encoder outputs/cross-attention state
to stay device-resident when stages are co-located.

### Iterative non-AR execution

Diffusion, flow matching, block-diffusion text generation and related models have
request state and repeated execution, but their scheduler unit may be timesteps,
latent shapes, guidance branches, resolution phases, or another model-specific
quantity rather than tokens.

They need their own runtime lifecycle and batching policy. Image/video pipelines
may compose text/image encoders, denoisers and VAE/media decoders. The top-level
orchestrator should compose these without teaching the AR scheduler about latents.

Additional runtime families can be added if real models demonstrate another common
execution regime. Do not invent a closed universal list now.

## Continuation resources and prefix reuse

For AR models, scheduling and state management must cooperate. One-time
`Ready/Deferred` admission plus a full maximum-context reservation is not sufficient
for high-concurrency serving.

The target resource layer should support dynamic reservation, per-step preparation,
commit/update, release, cache lookup/reuse, eviction and preemption. The scheduler
needs capacity/cost information, while physical tensor/page/checkpoint layouts
remain model/backend-owned.

A semantic prefix may have several continuation components. SGLang's Unified Radix
Cache is a useful reference: one logical prefix tree can coordinate Full, SWA and
Mamba components while each component owns its allocation/eviction details and a
prefix match advances only when all required validators pass. Ribn does not need
its exact tree or page layout, but it should preserve that invariant.

For hybrid models such as the current Qwen path, reusable full-attention KV without
the matching recurrent state is not a valid continuation. Recurrent checkpoints,
copy-on-write/materialization cost, and cache pressure are first-class resource
concerns.

## Host/device execution design

The Rust host path should be asynchronous-first and data-oriented:

- one owner for mutable execution state; cloneable handles submit bounded commands;
- stable/generational request slots for the active lifetime;
- persistent per-request host/device rows with incremental metadata updates;
- avoid rebuilding large batch metadata every step;
- caller/runtime-owned reusable output/workspace buffers rather than hot-path
  allocations where profiling shows value;
- pinned/staged updates or device-native metadata preparation where appropriate;
- GPU completion via events/async observation instead of synchronization in the
  normal serving loop;
- graph capture and compiled/fused execution as explicit qualified variants;
- backend-specific physical representations rather than a lowest-common-denominator
  tensor/device abstraction.

vLLM Model Runner V2's permanent request rows, staged metadata writes,
async-first execution and GPU-native metadata preparation are particularly relevant.
FlashInfer's packed/ragged metadata, caller-owned workspaces and mixed
prefill/decode attention reinforce the same direction.

Coarse dynamic dispatch at model/stage boundaries is acceptable; per-layer or
per-token virtual dispatch in hot kernels is not the target. Rust's ownership/type
system should make resource lifetime and state transitions easier to prove without
turning every operation into a trait hierarchy.

## Model support velocity

Ribn needs a deliberate answer for new architectures.

Native support should make architecture differences explicit and reuse proven
components: normalization, rotary position handling, attention families, MoE,
convolutions/SSM operators, sampling, quantized linears, encoders, diffusion blocks,
media processors, and distributed collectives. New architecture code should compose
these pieces while retaining freedom to specialize hot paths.

Research a compatibility/reference backend separately. Candidate directions include
an optional Transformers/PyTorch bridge for correctness/bring-up, graph import from
a portable exported representation where coverage is adequate, or reuse of a Rust
framework only where it does not constrain Ribn's production execution design.
Do not pick one before measuring model coverage, fidelity, operator gaps and
maintenance cost.

Model qualification should compare against an independent reference artifact and
implementation. A parsed config or successful load is not execution support.

The contribution boundary is model-local for architectures using existing execution
mechanisms: model/processor/backend code, registration and tests. Cancellation,
output routing, protocol adapters and unrelated scheduler policy should not acquire
model-family branches. Genuinely new state mechanisms or execution regimes can require
focused core changes; a frozen core is not the goal. The roadmap's model-local
integration gate requires a second real decoder and a real VLM to demonstrate this
boundary before it is advertised as routine.

Optimized execution variants may use different arithmetic and physical layouts.
Keep exact checks for transformations intended to preserve arithmetic, and separately
qualify reordered algorithms against justified numerical criteria. Existing baseline
error is not a budget for another optimization, and token agreement on one prompt is
not sufficient qualification. The recurring procedure lives in
[the model-integration skill](../.agents/skills/model-integration/SKILL.md).

## Hardware and distributed execution

The development RTX 4090 is a qualification device, not an architecture boundary.
Ribn should permit backend-specific NVIDIA, AMD, Apple/Metal and CPU implementations
without requiring identical kernels or memory layouts.

Ribn may own inference-local parallelism and communication: tensor, expert,
pipeline, data/context/sequence parallel strategies; intra-model collectives;
prefill/decode disaggregation; stage-local replica execution; and model-state
transfer. External systems such as Archon/Kubernetes/Slurm can allocate devices and
place processes. Ribn should accept those resources without depending on a fleet
manager.

Logical model topology and deployment topology must stay distinct. A stage can be
co-located today and placed on another device/node later without changing the model
semantics.

## Expected use cases

The design should support, progressively rather than all at once:

- embedded/local Rust applications and CLI use;
- offline batch inference/evaluation;
- Python in-process inference and RL/post-training rollout workloads;
- OpenAI/Anthropic-compatible serving and native extensions where useful;
- multimodal understanding with image/audio/video inputs;
- embedding, classification, scoring and reranking;
- ASR/translation and speech/audio generation;
- image and video generation/editing;
- structured output, tools/reasoning parsing and speculative generation;
- adapters/multi-LoRA and quantized models;
- single accelerator, multi-accelerator and multi-node inference;
- eventually multi-model serving, sleep/wake/weight-update workflows and
  disaggregated inference.

Supporting a use case means implementing and qualifying it; listing it here only
means the architecture must not make it require a foundational rewrite.

## External design lessons

The comparisons below are inherited research leads, not freshly verified feature
claims or accepted implementation choices. Verify current primary source and pin
its revision before using a volatile external behavior to decide a contract. Retain
that evidence with the decision; do not copy another engine's process structure
without measuring the need in Rust.

| Engine | What to borrow | Main trade-off / warning |
| --- | --- | --- |
| vLLM V1/MRV2 | Unified token scheduling, persistent request rows, async-first metadata/execution, broad APIs, Transformers fallback | Historically AR/text-centered and complex enough to require major redesigns; omni execution now lives in an extension |
| vLLM-Omni | Logical stage graph, specialized AR/diffusion runtimes, separation of model topology from deployment placement, typed multimodal transport | Young and still evolving; stage architecture can become excessive if every logical boundary becomes a process/layer |
| SGLang | Radix/prefix-cache co-design, hybrid Full/SWA/Mamba cache components, aggressive distributed/PD/EP execution, structured generation, diffusion | Extremely fast-moving and complex; its own omni work is removing layers, which argues for a shallow Ribn control plane |
| TensorRT-LLM | Scheduler/resource-manager cooperation, deep NVIDIA optimization, KV transfer/disaggregation, parallelism | NVIDIA-centric and a large moving surface; recent removal of the old TensorRT engine backend shows backend architecture can change substantially |
| MAX | Architecture registry/package, HF+safetensors/GGUF loading, shared serving stack, compiler/kernel specialization, NVIDIA/AMD/Apple and diffusion/video support | A compiler/runtime ecosystem is a major investment; Ribn should not build a compiler merely to imitate it |
| llama.cpp | Local/edge portability, simple direct APIs, quantization/GGUF, multimodal processor separation, low deployment friction | Manual architecture integration and GGUF-centric workflows; less aligned with large distributed datacenter serving and non-LLM media generation |
| mistral.rs | Rust-native cloneable model SDK, engine-thread/channel ownership, text/multimodal/speech/diffusion/embedding builders, paged attention | Smaller ecosystem and distributed-serving evidence than vLLM/SGLang/TRT-LLM; broad support is maintained through many model pipelines and Candle-based components |
| MLC LLM | Compilation for cross-platform deployment including mobile/browser; generated hardware-specialized model libraries | Compilation/model-conversion complexity and a large compiler scope that Ribn does not currently need |
| LMDeploy | Persistent batching, cache manager, current SSM checkpoint/prefix-cache controls and quantized serving | Multiple execution backends and a historically NVIDIA/TurboMind-centered architecture add complexity |
| TGI | Rust router, explicit protocol/execution split, production telemetry/backpressure | Maintenance mode since 2025; useful architectural history, not the feature target |

## Current Ribn: preserve, redesign, replace

### Preserve

- explicit RequestId vs SequenceId ownership;
- whole-batch validation before logical commitment;
- cancellation that never frees in-flight device state early;
- bounded input/output and per-request backpressure;
- conservative fault/shutdown resource ownership;
- architecture config separated from GGUF parsing;
- model/backend-specific physical resources and kernels;
- independent numerical reference tests and qualification gates.

### Redesign

- demote `GenerationExecutor` and the current `Engine` to the AR execution domain;
- replace the idea that encoded token generation is the top-level engine API;
- add a loaded-model/model-package boundary and a shallow cross-stage orchestrator;
- replace Qwen/GGUF/CUDA-specific `TextModel::load` with model-source resolution and
  architecture registration;
- add first-class Hugging Face/safetensors/tokenizer/processor loading;
- redesign AR resource admission around dynamic continuation resources;
- revisit separate prefill/decode scheduling once those resources exist;
- make the high-level `Model` handle concurrent/async instead of `&mut TextModel`;
- allow non-token results and multimodal prepared inputs without leaking them into
  token-scheduler internals.

### Retire after migration

- the duplicate legacy serving runtime in `engine-core`;
- Qwen-specific frontend assembly and any supposedly generic type whose only
  implementation is actually Qwen/GGUF/CUDA;
- full-context-per-sequence reservations as the normal serving memory model.

## Architecture pressure tests before deeper build-out

Before declaring the new top-level contracts stable, exercise them with materially
different small/reference-backed paths:

| Pressure test | What it must prove |
| --- | --- |
| Current Qwen hybrid AR | Full + recurrent continuation state, quantization, chunked AR execution, cancellation |
| Dense decoder-only model | Model package reuse without Qwen-specific scheduler changes |
| Encoder/pooling model | Embedding/scoring works without fake AR sequences |
| Encoder-decoder ASR model | Encoder output can feed iterative/AR decoding without token-only top-level assumptions |
| VLM | Image/video processor + encoder + AR path; model-specific placeholder/position semantics stay outside common scheduler |
| Diffusion image/video model | Non-token iterative runtime and media output work without branching the AR scheduler |
| Non-AR text model if practical | Text output does not imply autoregressive execution |
| Second hardware backend | Generic lifecycle/orchestration survives a materially different device/runtime implementation |

Full production support for all of these is not required before work continues.
Small reference-backed implementations are sufficient to expose wrong boundaries.

## Non-decisions

This design does not commit Ribn to vLLM's scheduler, SGLang's radix tree,
vLLM-Omni's exact process structure, a universal tensor IR, one page size, a Python
production dependency, a general ML compiler, or a dynamic Rust plugin ABI.

It does commit Ribn to treating the current Qwen/4090 path as a test and
qualification vehicle rather than the scope of the engine.
