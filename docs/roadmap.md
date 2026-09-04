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

Phase 4 begins with a boundary correction before scheduler work grows around Phase-3-specific assumptions.

### 4A — generic runtime state and resource boundaries

- Replace Qwen-shaped runtime state containers with generic typed `InferenceState` / `InferenceStateSet` boundaries while preserving backend-specific Qwen physical state.
- Keep inference-state identity/lifecycle separate from model residency and weight-placement concerns.
- Remove temporary compatibility names once current CUDA callers migrate.
- Keep model regions and execution phases narrow; do not introduce a general compiler IR.

### 4B — persistent request runtime

- Define stable active-request slots and request lifecycle transitions.
- Store persistent scheduling/model metadata once and update it incrementally.
- Avoid rebuilding large request/batch metadata structures every iteration.
- Make cancellation, completion, error, and reclamation ownership explicit.

### 4C — async-first execution loop

- Define host/device ownership and N/N+1 overlap rules.
- Eliminate unnecessary synchronization from the steady-state decode path.
- Preallocate/reuse step metadata and device-side buffers where practical.
- Add explicit readiness/warmup states rather than treating process startup as semantic readiness.

### 4D — continuous batching

- Admit/wait/run requests under explicit work/token budgets.
- Form mixed prefill/decode work without hard-coding one immutable request topology.
- Establish fairness/priority hooks without putting expensive policy search in the fast loop.
- Benchmark throughput, TTFT, ITL, tail latency, and host overhead against incumbent configurations with matched semantics.

### 4E — state paging and reuse

- Add block/paged KV allocation where it improves real workloads.
- Preserve recurrent/other required state at reusable prefix boundaries.
- Add exact prefix reuse only when the complete model-required state can be reconstructed correctly.
- Make allocation/reuse/transfer costs visible to scheduling.

### 4F — minimal serving surface

- Serve the same runtime used by direct/local inference.
- Streaming request/response lifecycle with cancellation and backpressure.
- OpenAI-compatible surface where useful without coupling core semantics to that protocol.
- Tokenization/detokenization and chat-template work runs outside the device hot path and is bounded under load.

## Phase 5 — performance system

- Execution-variant registry with explicit compatibility/qualification identity.
- CUDA graph/capture paths only after eager-path correctness gates exist for the same semantics.
- Fused/vendor/custom kernel selection based on measurements.
- Persistent and larger execution regions where they beat simpler paths.
- Runtime profiling and empirical cost models.
- Versioned live policy with safe application boundaries and rollback.
- Transparent cache/preparation identity for packed weights, compiled kernels, graphs, profiles, and warmup products.

The common path should first improve by removing host work, allocations, copies, and synchronization before relying on complex tuning.

## Phase 6 — speculation

- Use Qwen3.8 native MTP as the first speculation path.
- Treat propose/verify/state/cost behavior as a provider/runtime capability, not a Qwen-specific scheduler API.
- Add acceptance/cost telemetry and adaptive policy only after ordinary decode is competitive and stable.
- Qualify speculation jointly with graph mode, quantization, state layout, and hardware before automatic selection.

## Phase 7 — broader model and hardware coverage

Prioritize architectures that force useful generalization rather than a long checklist of similar dense decoders.

- Additional dense/hybrid model families.
- MoE/expert execution and sparse-attention architectures.
- Multimodal/encoder stages.
- Newer NVIDIA generations and materially different CUDA capabilities.
- Metal and AMD when the backend/runtime contracts are mature enough to test portability honestly.
- Additional checkpoint/quantization formats based on user value.

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

No novelty claim is required for the project to be useful. Near-term success is stronger execution across performance, latency, memory efficiency, startup, portability, correctness, observability, configuration, and usability.
