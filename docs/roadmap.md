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

The architecture correction and first correctness serving loop are now in place. Core interfaces do not encode the Phase-3 Qwen state bundle, and the same runtime can drive persistent scheduled requests through the CUDA Qwen executor and expose committed output to a frontend.

The RTX 4090 remains the available local qualification machine, not a long-term architecture target.

### 4A — runtime boundary correction — complete

- Generic typed `InferenceState` / `InferenceStateSet` at core runtime/backend boundaries.
- Backend-specific physical Qwen state remains free to be specialized.
- Model residency is distinct from per-request inference state.
- Model regions remain distinct from execution phases; no general compiler IR was introduced.
- Temporary core `HybridState*` compatibility aliases and NVIDIA callers were migrated.
- Execution variants carry explicit qualified/experimental/incompatible status.
- Runtime startup has explicit readiness states.
- Qwen recurrent state now describes actual persistent matrix-bank and convolution-history storage rather than model projection-head geometry.

### 4B — persistent request/runtime foundation — complete

- Stable generation-tagged active-request slots.
- Explicit waiting/runnable/in-flight/terminal lifecycle states.
- Persistent request progress and inference-state ownership.
- Submission identity prevents a completion from updating the wrong request.
- Core execution uses submit/poll completion semantics; logical state commits only after completion.
- Backend physical state is keyed by stable logical state identity and explicitly released before logical reclamation.

### 4C — scheduler and serving loop — in progress

Implemented correctness foundation:

- Deterministic admission and persistent runnable/in-flight/terminal bookkeeping.
- Explicit token budgets with decode priority and chunked prefill.
- Multi-request `ExecutionBatch` submission through the backend boundary.
- Prompt/decode token-input ownership across iterations.
- Committed generated-token delivery separate from next-decode-input state.
- Cancellation of an in-flight request waits for backend completion, suppresses its output, and does not cancel peers in the same submission.
- Submission failure, completion, terminal reclamation, logical-state release, and physical CUDA-state release are tested lifecycle paths.
- The Qwen CUDA serving dispatcher preserves the scheduler's whole-batch boundary and persistent physical state.

Still required before 4C is complete:

- Replace the current sequential batch-1 CUDA compatibility execution with a native batched path where measurements justify it.
- Replace synchronous CUDA completion behavior with genuinely asynchronous submission/completion; no mandatory host synchronization should remain in the steady-state token loop.
- Reuse/preallocate batch metadata and device-side step buffers where measurements justify it.
- Benchmark scheduler CPU overhead, TTFT, ITL, tail latency, and throughput against matched incumbents.
- Qualify failure/cancellation behavior on the real asynchronous CUDA path, not only the current correctness dispatcher.

### 4D — state paging and exact reuse

- Add block/paged KV allocation where it improves real workloads.
- Preserve recurrent/other required state at reusable prefix boundaries.
- Add exact prefix reuse only when the complete model-required state can be reconstructed correctly.
- Make allocation/reuse/transfer costs visible to scheduling.
- Add preemption only with explicit state ownership/reclamation semantics.

### 4E — minimal serving surface

A direct local Qwen command now exercises tokenizer → scheduler → CUDA runtime → generated-token delivery → detokenizer over the same serving runtime. It is a correctness/qualification frontend, not completion of the network serving surface.

Remaining work:

- Streaming request/response lifecycle with cancellation and bounded backpressure.
- OpenAI-compatible surface where useful without coupling core semantics to that protocol.
- Keep tokenization/detokenization and chat-template work outside the device hot path and bounded under load.
- Readiness must reflect required preparation/warmup rather than process liveness.

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
