# Architecture

Ribn is a model inference runtime and serving engine. Its long-term goal is broad model and hardware coverage for individual developers through research labs, with execution competitive with or better than established engines. These are goals, not claims about current coverage or performance.

The core design goal is to make inference a live, stateful runtime problem rather than a fixed startup configuration.

## Scope

Engine owns model loading and execution, request execution, scheduling and batching, inference-state lifecycle, device backends, execution variants, runtime policy, profiling, and eventually distributed inference. It does not own fleet provisioning, global datacenter scheduling, training, or a general-purpose tensor compiler.

Local/offline and server frontends share the same semantic runtime.

```text
frontend / local API
        ↓
request state + policy
        ↓
request scheduler
   ↙           ↘
state       execution plan
manager         ↓
   ↘       execution runtime
      ↘       ↓
        model provider
             ↓
       compute backend
             ↓
 execution variants / hardware
```

## Runtime invariants

### Persistent request state

Active requests occupy stable generation-tagged runtime slots. Prompt progress, generated progress, lifecycle, and inference-state ownership persist across steps rather than being reconstructed from transient batches.

The scheduler will gather work from these slots and update only what changes. Cancellation, failure, completion, and reclamation are explicit lifecycle transitions.

### Async-first execution

The core backend contract is submission/completion based. Submitting a segment does not imply host synchronization with device completion, and logical inference-state position is committed only after completion becomes visible.

This lets a serving loop prepare step N+1 while device work for step N is in flight when dependencies permit it. The NVIDIA serving dispatcher enqueues kernels and pinned output copies, then polls completion events. Its eager path remains an explicit correctness oracle.

### Typed semantic inference state

Inference state is not synonymous with KV cache. The core models state as typed families collected at a semantic prefix boundary. Current families include full-attention KV and recurrent/linear-attention state. New state families should be added only when a real model requires them, without changing scheduler/backend container signatures.

A state requirement describes the information the model needs to continue correctly at a semantic prefix. Physical encoding, precision, packing, layout, placement, paging, compression, reconstruction, and distributed sharding are execution/backend concerns. Two qualified execution variants may realize the same semantic state differently without creating different semantic state families.

Strong semantic types are intentional. Do not replace them with an open-ended tensor map or universal state IR merely to accommodate future architectures.

The current Qwen path still carries concrete storage dtypes in `KvStateSpec` and `RecurrentStateSpec`, and current logical capacity accounting depends on those fields. Treat that as a transitional implementation constraint rather than the intended semantic identity. The state-boundary refactor should remove this coupling before the next major model architecture is implemented, after the current CUDA Rust migration proof gates unless the migration would otherwise deepen the wrong contract.

Packed or sub-byte state encodings with block/group scaling metadata should not be forced into an ordinary scalar API whose fundamental operation is whole-byte `byte_width()`. Compute scalar types and physical state-storage encodings may remain distinct concepts when a real implementation requires that split.

### Prefix coherence does not imply co-location

Every state family in one request state set represents the same semantic prefix boundary. That invariant is semantic; it does not require every physical component to occupy one device or storage tier.

A qualified execution variant may eventually combine device-resident, host-resident, paged, sharded, transferred, or reconstructable components. Placement compatibility belongs to backend/variant validation.

The current `InferenceStateSet` still enforces one `StateLocation` for the complete set. This is a temporary implementation restriction and should be removed in the scheduled state-boundary refactor. Do not add paging or tiering merely to justify the type change.

### Materialization is explicit

A nonzero semantic prefix is valid only when semantically correct continuation state exists for that prefix. Empty allocation never constitutes restoration.

A backend may eventually satisfy a prefix by retaining resident state, transferring/restoring state, reconstructing it with a bounded qualified procedure, or reusing an exact compatible prefix. Those are materialization mechanisms, not changes to semantic request identity. Ownership, cost, and completion semantics must remain explicit.

### Inference state is not model residency

Request state and model residency are separate resource planes. KV/recurrent/draft state follows request semantics and prefix lifecycle. Weights and other model-owned resources may be device-resident, host-resident, mapped, sharded, prefetched, or packed independently.

`ModelResidencyPlan` captures this distinction without turning model resources into request state or a generic object store. Read-mostly external lookup memories, expert resources, packed weights, compiled kernels, and similar model-owned assets remain residency concerns even when they live in host memory or require asynchronous prefetch.

Residency should evolve from concrete model requirements. Do not turn it into a general storage service, automatic tiering daemon, or arbitrary artifact system before a real model path needs those mechanisms.

### Request phase is distinct from model structure

Model providers expose inference-oriented regions and requirements. Model regions/stages describe model structure; request execution phases describe serving work that matters to scheduling. They are deliberately separate concepts.

Prefill, decode, and speculative propose/verify can be request phases when their scheduling and state-transition behavior differs. Encoder blocks, MoE experts, sparse-index work, multimodal towers, and similar structures are normally model regions or execution stages inside a request phase rather than global scheduler phases.

The current `ExecutionPhase` enum still contains `Encoder` and `MoEExpert` from the early boundary design. Do not build additional scheduler behavior around those variants. Clean up the taxonomy before broader model support unless a concrete model proves one of them is genuinely a top-level request lifecycle phase.

### Narrow execution description

The core does not attempt to be MLIR/TVM/LLVM or an arbitrary tensor compiler. Execution plans contain only information the runtime needs to prepare and execute inference efficiently.

Model providers should describe the stages, semantic state, capabilities, and model-owned resources needed by inference. Hardware/backend code owns device-specific mechanics, kernels, layouts, transfers, and execution choices.

### Capability-driven backends

`ComputeBackend` is a coarse semantic boundary: validate, submit, and observe completion. The core owns common request/state/plan semantics; a backend owns device resources, runner mechanics, streams/queues, graph capture, transfers, collectives, and hardware-specific execution choices.

Inside a backend, share behavior when semantics are genuinely common and keep replaceable hardware mechanisms and kernel/variant selection target-specific. Do not create a universal backend-component framework before a second backend exposes real duplication.

Portability does not mean one lowest-common-denominator kernel stack.

CUDA Rust is the intended NVIDIA implementation foundation: cuTile for tile-oriented kernels and cuda-oxide for explicit SIMT control. Its compiler, tensor, and asynchronous execution types stay inside the NVIDIA backend. Adoption preserves the semantic submit/completion and resource-lifecycle contracts; it does not imply a performance or qualification claim. See the [migration design](cuda-rust-migration.md) for the proof gates and retirement of the existing CUDA C++ authoring pipeline.

### Qualified execution variants

Execution variants carry an explicit qualification status:

- `Qualified`: eligible for automatic selection;
- `Experimental`: available only through an explicit choice or experiment;
- `Incompatible`: rejected for the prepared plan.

Qualification is scoped to the compatibility identity that matters for correctness: model/revision, hardware/runtime, quantization, physical state representation and encoding, graph/capture mode, speculation mode, and distributed layout as applicable. Supporting each feature individually is not evidence that their combination is correct.

Semantic state requirements should not absorb those representation choices merely so qualification can distinguish them. Qualification identity is the correct place to bind a semantic model requirement to one concrete physical execution realization.

### Phase-specific cost is planning input, not scheduler ontology

Model architectures can have materially different prefill and decode compute, state-materialization costs, resource use, and execution variants. The runtime should be able to measure and identify those differences without teaching the common scheduler model-specific layer math.

Future empirical cost inputs can include request phase, selected execution variant, state transfer/materialization cost, model-resource prefetch, batch shape, speculation behavior, hardware, and distributed layout. Expensive profiling, search, compilation, and autotuning stay off the per-step critical path.

### Cheap fast scheduler

The request scheduler operates from persistent request slots, available work/token budget, inference-state availability, and a current policy snapshot. Expensive search, profiling, compilation, or autotuning stays off the per-step hot path.

The initial scheduler should be deterministic and simple: admit work, prioritize latency-sensitive decode according to policy, use remaining budget for chunked prefill, submit work asynchronously, and reclaim completions. More sophisticated policy must earn its complexity in measurements.

### Transparent preparation and readiness

Packed weights, compiled/JIT kernels, graph products, profiles, and similar artifacts are derived runtime assets, not a mandatory user-visible compile ceremony. Preparation should normally be transparent and cached; explicit preparation can exist for deterministic deployment later.

Runtime readiness is explicit: created, loading, preparing, optional warming, then ready. A live process is not necessarily able to serve. Required execution paths should be prepared or have a qualified safe fallback before readiness is reported.

The prepared-artifact cache itself is later performance-system work; the current architecture establishes its identity/readiness boundary without putting it in the Phase 4 scheduler critical path.

## Model-provider boundary

Model providers describe model regions/stages, semantic state requirements, capabilities, tokenizer/prompt behavior, model-owned resource requirements, and weight sources without teaching the scheduler architecture-specific layer math.

The first native path is Qwen3.8-27B. Compatibility providers may later use external ecosystems where that is the fastest route to model coverage, but Python or another framework must not own the request scheduler hot path.

Future multimodal support should preserve this boundary: request/front-end semantics describe multimodal inputs; the model provider owns model-specific preprocessing and encoder requirements; the execution plan exposes the stages the runtime must execute. Do not encode non-text inputs as fake text tokens merely to preserve a frontend shape.

## Hardware boundary

The RTX 4090 is the available initial qualification and development machine. It is the first testable target within the RTX 3090-and-up class; the architecture does not depend on that device or its 24-GiB memory limit.

Future NVIDIA generations, AMD, Metal, and other devices should pressure-test the same semantic contracts while using target-specific execution mechanisms.

## Distributed boundary

Distributed inference is eventually an Engine runtime capability: tensor/expert/pipeline/sequence parallelism, distributed model residency and request state, inference-local routing, and disaggregation where justified by measurements.

Physical machine allocation, fleet health, and datacenter-wide scheduling remain external. Ribn must remain independently usable; a fleet manager such as Archon can consume execution/resource options later without becoming a required dependency.

## Optimization stance

The common path should first become fast by removing avoidable work: stable slots, preallocation, incremental metadata, asynchronous submission, suitable state layouts, fewer copies, and fewer synchronization points. Complex policy, compilation, and autotuning are layered on top of measured costs rather than used to compensate for avoidable runtime overhead.

New abstractions should be earned by concrete implementation pressure. Generalize semantic ownership and lifecycle boundaries early; generalize hardware/model mechanisms only after real duplication proves the abstraction.
