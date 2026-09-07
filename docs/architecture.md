# Architecture

Engine is a model inference runtime and serving engine. The core design goal is to make inference a live, stateful runtime problem rather than a fixed startup configuration.

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

### Typed inference state

Inference state is not synonymous with KV cache. The core models state as typed families collected at a semantic prefix boundary. Current families include full-attention KV and recurrent/linear-attention state. New state families should be added only when a real model requires them, without changing scheduler/backend container signatures.

Logical identity and lifecycle stay separate from backend-owned physical layout and storage. Backend-specific physical structures may remain model-specialized where that is useful.

### Inference state is not model residency

Request state and model residency are separate resource planes. KV/recurrent/draft state follows request semantics and prefix lifecycle. Weights and other model-owned resources may be device-resident, host-resident, mapped, sharded, prefetched, or packed independently.

`ModelResidencyPlan` captures this distinction without turning model resources into request state or a generic object store.

### Narrow execution description

Model providers expose inference-oriented regions and requirements. Model regions describe model structure; execution phases describe serving work. They are deliberately separate concepts.

The core does not attempt to be MLIR/TVM/LLVM or an arbitrary tensor compiler. Execution plans contain only information the runtime needs to prepare and execute inference efficiently.

### Capability-driven backends

`ComputeBackend` is a coarse semantic boundary: validate, submit, and observe completion. The core owns common request/state/plan semantics; a backend owns device resources, runner mechanics, streams/queues, graph capture, transfers, collectives, and hardware-specific execution choices.

Inside a backend, share behavior when semantics are genuinely common and keep replaceable hardware mechanisms and kernel/variant selection target-specific. Do not create a universal backend-component framework before a second backend exposes real duplication.

Portability does not mean one lowest-common-denominator kernel stack.

### Qualified execution variants

Execution variants carry an explicit qualification status:

- `Qualified`: eligible for automatic selection;
- `Experimental`: available only through an explicit choice or experiment;
- `Incompatible`: rejected for the prepared plan.

Qualification is scoped to the compatibility identity that matters for correctness: model/revision, hardware/runtime, quantization, state representation, graph/capture mode, speculation mode, and distributed layout as applicable. Supporting each feature individually is not evidence that their combination is correct.

### Cheap fast scheduler

The request scheduler operates from persistent request slots, available work/token budget, inference-state availability, and a current policy snapshot. Expensive search, profiling, compilation, or autotuning stays off the per-step hot path.

The initial scheduler should be deterministic and simple: admit work, prioritize latency-sensitive decode according to policy, use remaining budget for chunked prefill, submit work asynchronously, and reclaim completions. More sophisticated policy must earn its complexity in measurements.

### Transparent preparation and readiness

Packed weights, compiled/JIT kernels, graph products, profiles, and similar artifacts are derived runtime assets, not a mandatory user-visible compile ceremony. Preparation should normally be transparent and cached; explicit preparation can exist for deterministic deployment later.

Runtime readiness is explicit: created, loading, preparing, optional warming, then ready. A live process is not necessarily able to serve. Required execution paths should be prepared or have a qualified safe fallback before readiness is reported.

The prepared-artifact cache itself is later performance-system work; the current architecture establishes its identity/readiness boundary without putting it in the Phase 4 scheduler critical path.

## Model-provider boundary

Model providers describe model regions, state requirements, capabilities, tokenizer/prompt behavior, and weight sources without teaching the scheduler architecture-specific layer math.

The first native path is Qwen3.8-27B. Compatibility providers may later use external ecosystems where that is the fastest route to model coverage, but Python or another framework must not own the request scheduler hot path.

## Hardware boundary

The RTX 4090 is the available initial qualification and development machine. It is the first testable target within the RTX 3090-and-up class; the architecture does not depend on that device or its 24-GiB memory limit.

Future NVIDIA generations, AMD, Metal, and other devices should pressure-test the same semantic contracts while using target-specific execution mechanisms.

## Distributed boundary

Distributed inference is eventually an Engine runtime capability: tensor/expert/pipeline/sequence parallelism, distributed model residency, inference-local routing, and disaggregation where justified by measurements.

Physical machine allocation, fleet health, and datacenter-wide scheduling remain external.

## Optimization stance

The common path should first become fast by removing avoidable work: stable slots, preallocation, incremental metadata, asynchronous submission, suitable state layouts, fewer copies, and fewer synchronization points. Complex policy, compilation, and autotuning are layered on top of measured costs rather than used to compensate for avoidable runtime overhead.
