# Roadmap

This roadmap describes implementation order, not a promise of release dates. Correctness gates precede performance claims.

## Completed foundations

### Phase 1 — target and incumbent baselines

- Pin the first Qwen3.8-27B artifact and semantics.
- Establish llama.cpp same-artifact reference behavior.
- Capture initial vLLM/SGLang comparison evidence where the available artifact paths permit it.
- Establish benchmark methodology and hardware qualification workflow.

### Phase 2 — core contracts and loading

- Typed model regions, request semantics, runtime policy, execution plans, and backend capabilities.
- Typed KV and recurrent inference-state requirements and lifecycle.
- GGUF metadata/tensor loading and Qwen3.8 model description.
- Encoded quantized weight staging and CUDA execution primitives.
- Tokenizer/prompt-policy groundwork for the first artifact.

### Phase 3 — correct single-request Qwen execution

- Cross-validated host references for Qwen3.8 recurrent/Gated-DeltaNet, full attention, and FFN semantics.
- CUDA primitives and physical hybrid-state storage.
- End-to-end text execution on the RTX 4090.
- Greedy generation parity against the pinned llama.cpp reference for a 200-token run.
- Initial batch-1 correctness-path timing.

Phase 3 proves a correct native execution path. It does not establish a competitive serving engine yet.

## Phase 4 — runtime and serving foundation

The pre-scheduler architecture correction is complete. Core interfaces no longer encode the Phase-3 Qwen state bundle, and the serving/runtime foundation now has stable request-slot, residency, qualification, readiness, and submission/completion contracts.

The RTX 4090 remains the available local qualification machine, not a long-term architecture target.

### 4A — runtime boundary correction — complete

- Generic typed `InferenceState` / `InferenceStateSet` at core runtime/backend boundaries.
- Backend-specific physical Qwen state remains free to be specialized.
- Model residency is distinct from per-request inference state.
- Model regions remain distinct from execution phases; no general compiler IR was introduced.
- Temporary core `HybridState*` compatibility aliases and NVIDIA callers were migrated.
- Execution variants carry explicit qualified/experimental/incompatible status.
- Runtime startup has explicit readiness states.

### 4B — persistent request/runtime foundation — complete

- Stable generation-tagged active-request slots.
- Explicit waiting/runnable/in-flight/terminal lifecycle states.
- Persistent request progress and inference-state ownership.
- Submission identity prevents a completion from updating the wrong request.
- Core execution uses submit/poll completion semantics; logical state commits only after completion.

These are contracts and data structures, not yet a complete serving scheduler.

### 4C — scheduler and async serving loop — next

- Integrate admission and `RequestSlots` into one deterministic scheduler loop.
- Maintain ready/runnable/in-flight sets without rebuilding request metadata each iteration.
- Schedule under explicit token/work budgets.
- Prioritize latency-sensitive decode according to policy and fill remaining budget with chunked prefill.
- Poll completions while preparing later work; do not introduce a mandatory per-step host/device synchronization point.
- Make cancellation, failure, completion, and state/resource reclamation complete and testable at every lifecycle state.
- Reuse/preallocate batch metadata and device-side step buffers where measurements justify it.
- Replace the current synchronous NVIDIA compatibility-dispatch behavior with genuinely asynchronous CUDA submission when the serving loop can consume it.
- Benchmark scheduler CPU overhead, TTFT, ITL, tail latency, and throughput against matched incumbents.

### 4D — state paging and exact reuse

- Add block/paged KV allocation where it improves real workloads.
- Preserve recurrent/other required state at reusable prefix boundaries.
- Add exact prefix reuse only when the complete model-required state can be reconstructed correctly.
- Make allocation/reuse/transfer costs visible to scheduling.
- Add preemption only with explicit state ownership/reclamation semantics.

### 4E — minimal serving surface

- Serve the same runtime used by direct/local inference.
- Streaming request/response lifecycle with cancellation and backpressure.
- OpenAI-compatible surface where useful without coupling core semantics to that protocol.
- Tokenization/detokenization and chat-template work stays outside the device hot path and is bounded under load.
- Readiness reflects required preparation/warmup rather than process liveness.

## Phase 5 — performance system

- Execution-variant registry keyed by explicit compatibility and qualification identity.
- CUDA graph/capture paths only after eager-path correctness gates exist for the same semantics.
- Fused/vendor/custom kernel selection based on measurements.
- Persistent and larger execution regions where they beat simpler paths.
- Runtime profiling and empirical cost models.
- Versioned live policy with safe application boundaries and rollback.
- Transparent preparation cache for packed weights, compiled/JIT kernels, graphs, profiles, and warmup products.
- Exact artifact identity/invalidation and safe fallback; no surprise first-request compilation stall on a path reported ready.

The common path should first improve by removing host work, allocations, copies, and synchronization before relying on complex tuning.

## Phase 6 — speculation

- Use Qwen3.8 native MTP as the first speculation path.
- Treat propose/verify/state/cost behavior as a provider/runtime capability, not a Qwen-specific scheduler API.
- Add acceptance/cost telemetry and adaptive policy only after ordinary decode is competitive and stable.
- Qualify speculation jointly with graph mode, quantization, state layout, and hardware before automatic selection.

## Phase 7 — broader model and hardware coverage

Prioritize architectures and devices that force useful generalization rather than a long checklist of similar dense decoders.

- Additional dense/hybrid model families.
- MoE/expert execution and sparse-attention architectures.
- Multimodal/encoder stages.
- Newer NVIDIA generations and materially different CUDA capabilities.
- Metal and AMD when the backend/runtime contracts are mature enough to test portability honestly.
- Additional checkpoint/quantization formats based on user value.

The second backend should validate the current coarse core/backend boundary. Factor shared backend internals only where real duplication appears; do not pre-build a universal backend component framework.

## Phase 8 — distributed inference

Only after the single-node runtime is competitive and observable:

- prepared TP/PP/SP/EP plans;
- distributed model residency and state movement;
- inference-local routing and state affinity;
- prefill/decode or other disaggregation where workload measurements justify it;
- topology-aware communication and placement within externally allocated resources.

Engine remains a standalone inference runtime; fleet allocation belongs outside the project.

## Research track

Research can proceed alongside implementation but does not block the roadmap unless evidence shows a required boundary is wrong.

Potential directions include joint optimization of scheduling, execution variants, state placement, speculation, and topology; broader persistent/mega-kernel execution; state compression/reuse techniques; and new hardware-aware compilation strategies.

No novelty claim is required for the project to be useful. Near-term success is strong execution across performance, latency, memory efficiency, startup, portability, correctness, observability, configuration, and usability.
