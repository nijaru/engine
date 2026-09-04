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

Active requests should occupy stable runtime slots where practical. Per-step work mutates or gathers from persistent state instead of rebuilding large request/batch structures from scratch.

### Async-first execution

The normal execution loop should avoid gratuitous CPU/GPU synchronization. CPU preparation for the next step should be able to overlap device execution of the current step when dependencies permit it.

### Typed inference state

Inference state is not synonymous with KV cache. The core models state as typed families collected at a semantic prefix boundary. Current families include full-attention KV and recurrent/linear-attention state; future model families may add sliding KV, encoder, draft/speculation, adapter, or other state without changing scheduler/backend signatures.

Logical identity and lifecycle stay separate from backend-owned physical layout and storage.

### Inference state is not model residency

Request state and model residency are separate resource planes. KV/recurrent/draft state follows request semantics and prefix lifecycle. Weights and model-owned resources may be device-resident, host-resident, sharded, prefetched, packed, or otherwise prepared independently.

### Narrow execution description

Model providers expose inference-oriented regions and requirements. The core does not attempt to be MLIR/TVM/LLVM or an arbitrary tensor compiler.

Execution plans may describe model regions, state dependencies, placement, batch/work dimensions, graph/capture regions, communication, and execution choices.

### Capability-driven backends

Backends advertise concrete capabilities. Hardware-specific paths may use vendor libraries, external kernels, custom kernels, JIT/AOT code, graph capture, or fused/persistent execution when those choices are qualified and measured.

Portability does not mean one lowest-common-denominator kernel stack.

### Qualified execution variants

An optimized path is eligible for automatic selection only when its semantic compatibility and correctness have been established for the relevant combination of model/revision, hardware/runtime, quantization, state representation, graph/capture mode, speculation mode, and distributed layout.

Unknown combinations must fall back safely or fail explicitly. Throughput is not a correctness test.

### Cheap fast scheduler

The request scheduler operates from persistent request state, available work/token budget, state availability, and a current policy snapshot. Expensive search, profiling, compilation, or autotuning stays off the per-step hot path.

### Transparent preparation

Prepared kernels, graphs, packed weights, profiles, and similar artifacts are caches/derived runtime assets, not a mandatory user-visible compile ceremony. Cache identity and invalidation must be exact enough to preserve correctness and reproducibility.

### Explicit readiness

Model readiness is staged. Process startup, weights mapped, device resources prepared, execution variants qualified/prepared, warmup complete, and semantic readiness are distinct states where they matter.

## Model-provider boundary

Model providers describe model regions, state requirements, capabilities, tokenizer/prompt behavior, and weight sources without teaching the scheduler architecture-specific layer math.

The first native path is Qwen3.8-27B. Compatibility providers may later use external ecosystems where that is the fastest route to model coverage, but Python or another framework must not own the request scheduler hot path.

## Hardware boundary

The first implementation path is NVIDIA single-GPU CUDA. The RTX 4090 is development and benchmark hardware, not an architectural constraint.

Future NVIDIA generations, AMD, Metal, and other devices should use the same capability-driven conceptual boundary with target-specific implementations.

## Distributed boundary

Distributed inference is eventually an Engine runtime capability: tensor/expert/pipeline/sequence parallelism, distributed model residency, inference-local routing, and disaggregation where justified by measurements.

Physical machine allocation, fleet health, and datacenter-wide scheduling remain external.

## Optimization stance

The common path should first become fast by removing work: stable slots, preallocation, incremental metadata, asynchronous enqueue, suitable state layouts, and avoiding synchronization. More complex policy and autotuning are layered on top of measured costs rather than used to compensate for avoidable runtime overhead.
