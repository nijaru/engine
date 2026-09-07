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

The architecture correction and first correctness serving loop are in place. Core interfaces do not encode the Phase-3 Qwen state bundle, and the same runtime can drive persistent scheduled requests through the CUDA Qwen executor and expose committed output to a frontend.

The RTX 4090 remains the available local qualification machine, not a long-term architecture target.

### 4A — runtime boundary correction — complete

- Generic typed `InferenceState` / `InferenceStateSet` at core runtime/backend boundaries.
- Backend-specific physical Qwen state remains free to be specialized.
- Model residency is distinct from per-request inference state.
- Model regions remain distinct from execution phases; no general compiler IR was introduced.
- Temporary core `HybridState*` compatibility aliases and NVIDIA callers were migrated.
- Execution variants carry explicit qualified/experimental/incompatible status.
- Runtime startup has explicit readiness states.
- Qwen recurrent state describes actual persistent matrix-bank and convolution-history storage rather than model projection-head geometry.

### 4B — persistent request/runtime foundation — complete

- Stable generation-tagged active-request slots.
- Explicit waiting/runnable/submitting/in-flight/cancelling/terminal lifecycle states.
- Persistent request progress and inference-state ownership.
- Submission identity prevents a completion from updating the wrong request.
- Core execution uses submit/poll completion semantics; logical state commits only after completion.
- Backend physical state is keyed by stable logical state identity and explicitly released before logical reclamation.

### 4C — scheduler and serving loop — in progress

Implemented correctness foundation:

- Deterministic bounded admission and persistent runnable/in-flight/terminal bookkeeping.
- Explicit token budgets with decode priority and chunked prefill.
- Multi-request `ExecutionBatch` submission through the backend boundary.
- Prompt/decode token-input ownership across iterations.
- Committed generated-token delivery separate from next-decode-input/model-state progress.
- Multi-chunk prefill semantics in which only the final prompt chunk requests output and the following decode begins at the exact prompt boundary.
- Cancellation of an in-flight request waits for backend completion, suppresses its output, and does not cancel peers in the same submission.
- Submission failure, completion, terminal reclamation, logical-state release, and physical CUDA-state release have explicit lifecycle paths.
- The Qwen CUDA serving dispatcher preserves the scheduler's whole-batch boundary and persistent physical state.
- Intermediate Qwen prefill tokens skip output normalization, the vocabulary projection, argmax, and host token readback when no sampled output is semantically required.
- The Qwen dispatcher now completes asynchronously through dispatcher-owned pinned host output slots and one CUDA completion event per submission: enqueue failures flush the stream and leave no hidden async ownership, and state released while queued work still references it is deferred to completion. The eager blocking path remains the correctness fallback.
- The NVIDIA adapter has a narrow dispatcher-owned asynchronous submit/poll seam keyed by `BackendSubmissionId`; synchronous/reference dispatchers retain their existing behavior.
- Terminal asynchronous completion failure is defined to end backend access to request state/resources before the runtime may reclaim them.
- Full-model async-vs-eager greedy parity against the pinned llama-server continuation passes on the RTX 4090, and identical serving sweeps before/after async adoption show unchanged ~1.8 tok/s aggregate: host-blocking removal alone does not move throughput while rows still execute sequentially through batch-1 kernels.
- Warp-cooperative quantized GEMV is landed and hardware-qualified: eight one-warp-per-row variants (coalesced lane-strided element runs, shuffle reduction) match the scalar oracle in per-family parity tests and reproduce the full 200-token llama-server greedy continuation in warp mode. A same-commit A/B serving sweep measures ~9.2 tok/s in warp mode vs 1.81 tok/s scalar (5.1x), flat across concurrency 1/2/4/8: the per-projection kernel was the bottleneck, and scheduler-batch members still execute sequentially through batch-1 kernels, which the remaining gap to the incumbent pace (~49.5 tok/s same artifact) attributes to. Warp is the executor default since `2d01c56` (user-approved on the measured evidence); scalar remains via `GemvMode::Scalar` / bench `--gemv=scalar`, and the scalar full-model replay pins its mode explicitly so oracle coverage survives the flip. Re-validated 44/44 on hardware after the flip.
- An nsys trace of warp-mode decode then showed the remaining single-thread correctness-first kernels dominating: `rms_norm` at 41.5% of GPU time (280 us per vector) and `argmax` at 4.75 ms per vocabulary scan. `2267630` replaced both (plus `l2_norm`) with one-block parallel reductions preserving the reference equations, tie, and NaN behavior; the CUDA suite re-passed 44/44 and the serving sweep moved to ~15.4 tok/s (1.68x over warp-only, 8.5x over the session baseline), still flat across concurrency because batch members remain sequential through batch-1 kernels.
- A second trace then showed `attn_score_gqa` (one thread per query head, 32 threads device-wide) at 11.3%. `9794181` gave each query head a full warp: coalesced F16 KV reads, shuffle-reduced dots, running-max softmax with exact rescaling, numerically identical to the reference. Suite re-passed 44/44 (165 s); the sweep reached ~20.3 tok/s (ITL 44 ms at c1), 11.2x the session baseline. Kernel-level work is now near exhausted per the trace: warp GEMV families are ~74% of remaining GPU time, so the next structural lever is native batched projections, as the sweep's flat-concurrency shape already indicated.
- Native batched execution is landed, hardware-qualified, and measured (`72e0908`-`b97e4d6`): batched warp GEMV (one warp per (row, member), consecutive warps sharing weight rows), batched embedding/norms/elementwise/rope/argmax, per-member state kernels over view-sliced rows, and a `CudaQwen35BatchDecode` executor running one launch per (layer, op) for every member. The dispatcher's submit seam picks it for multi-row all-decode greedy batches and falls back to the per-row path otherwise, with member physical states taken out of the registry for the step and re-inserted on every path so no device state is lost; per-member greedy tokens land in the same pinned slots through the existing completion-event machinery. Hardware qualification (51/51 CUDA suite incl. executor-level, dispatcher-level, real-geometry view-wrapper, and four-layer-prefix parity gates) caught and fixed three batched-path bugs: GDN q/k consumed never-written scratch, batched rope took dims instead of pairs, and `kv_append_f16_views` covered only 64 of 1024 cache elements at real geometry. The serving sweep moved concurrency-8 aggregate from a flat 20.3 tok/s to 25.0 tok/s (c1 unchanged at 20.2 tok/s / 44 ms ITL), the first concurrency win of the project; the remaining batched-step cost (~6.25x one member for 8) points at weight re-reads per member that the one-warp-per-(row, member) GEMV expects L2 to dedupe.

Still required before 4C is complete:

- Close the batched-step cost gap: one batched 8-member step costs ~6.25x a single step because each (row, member) warp re-reads the full weight matrix; consecutive warps sharing rows rely on L2 dedupe, which the ~72 MB L2 cannot fully provide against ~150 MB FFN tensors. A weights-read-once GEMV variant (one warp reading each weight row, accumulating against all members' inputs in registers) is the evidence-pointed next kernel.
- Benchmark scheduler CPU overhead, TTFT, ITL, tail latency, throughput, GPU utilization, and memory against matched incumbents; llama.cpp holds ~49.5 tok/s at c1 on the same artifact.
- Qualify cancellation/failure behavior on the real asynchronous CUDA path, not only the eager correctness dispatcher.
- Benchmark scheduler CPU overhead, TTFT, ITL, tail latency, throughput, GPU utilization, and memory against matched incumbents.
- Qualify cancellation/failure behavior on the real asynchronous CUDA path, not only the eager correctness dispatcher.

Current deterministic decode-first policy can theoretically starve prefill if decode work continuously consumes the entire work budget. Treat bounded fairness as a measured scheduler-policy issue; add the smallest deterministic mechanism only if real workloads require it.

### 4D — state paging and exact reuse

A defensive validity guard is already present: fresh CUDA physical-state materialization at a nonzero logical prefix is rejected rather than allocating zero history and pretending it represents that prefix.

Remaining work:

- Implement real nonzero-prefix physical-state restoration/materialization.
- Add block/paged KV allocation where it improves real workloads.
- Preserve recurrent/other required state at reusable prefix boundaries.
- Add exact prefix reuse only when the complete model-required state can be reconstructed correctly.
- For Qwen3.8, reuse/restore must include both full-attention KV and recurrent/Gated-DeltaNet matrix plus convolution history at the same semantic prefix.
- Make allocation/reuse/transfer costs visible to scheduling.
- Add preemption only with explicit state ownership/reclamation semantics.

### 4E — minimal serving surface

A direct local Qwen command exercises tokenizer → scheduler → CUDA runtime → generated-token delivery → detokenizer over the same serving runtime. It is a correctness/qualification frontend, not completion of the network serving surface.

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
