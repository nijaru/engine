# Inference engine design

Date: 2026-09-12 (America/Los_Angeles)
Status: target architecture; current Qwen/CUDA implementation is a prototype subsystem

## Decision

Ribn is a general model inference engine. The current token-generation runtime is
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
