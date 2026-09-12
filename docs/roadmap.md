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

## Current implementation priority — CUDA Rust NVIDIA migration

The intended NVIDIA foundation is CUDA Rust, using cuTile and cuda-oxide as complementary kernel-authoring tracks. Execute the [migration design](cuda-rust-migration.md) before resuming unrelated kernel optimization: reproducible build/resource interoperability → representative quantized and recurrent kernels → asynchronous serving proof → complete coverage → qualification and legacy pipeline retirement.

This changes the NVIDIA implementation direction, not the semantic runtime boundaries or the remaining serving roadmap. The current CUDA C++ path remains the migration oracle until equivalent coverage is qualified; it is not intended as a permanent parallel implementation. Other hardware backend selection is deferred until a concrete implementation target requires it.

Do not interrupt these proof gates with speculative framework work. Core changes should move ahead during the migration only when the migration would otherwise lock in a known-wrong semantic contract.

## Architecture hardening before the next model family

The current high-level runtime boundary remains the intended direction, but several first-Qwen implementation details are not long-term semantic invariants. After the CUDA Rust proof sequence is sufficiently stable, and before native implementation of the next major architecture family, make one narrow state/execution-boundary pass:

- Separate semantic continuation-state requirements from physical dtype/encoding/layout. Current `KvStateSpec` / `RecurrentStateSpec` storage dtypes and semantic `byte_size()` accounting are transitional.
- Remove set-wide `StateLocation` equality from `InferenceStateSet`; preserve one coherent semantic prefix while allowing a qualified backend variant to validate per-component placement.
- Keep packed/sub-byte/block-scaled physical state encodings outside ordinary whole-byte scalar semantics. Do not add an `F4` variant merely to preserve `byte_width()`.
- Clean up request lifecycle phase versus model structure. `Encoder` and `MoEExpert` should not accumulate scheduler semantics merely because they exist in the current `ExecutionPhase` enum.
- Preserve exact qualification identity for the selected physical state realization rather than putting representation choices into semantic state families.
- Add tests that prove semantic identity remains stable across distinct physical representations once a second real representation exists, mixed placement is rejected or accepted by the selected backend rather than by the semantic container, and lifecycle/release ownership remains exact.

Do not build paging, automatic tiering, a universal state IR, or a generalized storage service as part of this pass. The goal is to remove false semantic assumptions, not to pre-implement every mechanism those types could eventually represent.

## Phase 4 — runtime and serving foundation

The architecture correction and first correctness serving loop are in place. Core interfaces do not encode the Phase-3 Qwen state bundle, and the same runtime can drive persistent scheduled requests through the CUDA Qwen executor and expose committed output to a frontend.

The RTX 4090 is the first testable hardware target within the RTX 3090-and-up class. Broader common hardware support remains the long-term direction.

### 4A — runtime boundary correction — complete

- Generic typed `InferenceState` / `InferenceStateSet` at core runtime/backend boundaries.
- Backend-specific physical Qwen state remains free to be specialized.
- Model residency is distinct from per-request inference state.
- Model regions remain distinct from execution phases; no general compiler IR was introduced.
- Temporary core `HybridState*` compatibility aliases and NVIDIA callers were migrated.
- Automatic variant eligibility requires variant-specific evidence matching the exact model, artifact, backend/device, runtime, and state/execution identity. Explicit baseline execution remains available without claiming automatic qualification.
- Runtime startup has explicit readiness states.
- Qwen recurrent state describes actual persistent matrix-bank and convolution-history storage rather than model projection-head geometry.

The semantic/physical state split listed above is a follow-up hardening pass, not a rollback of these completed ownership/container boundaries.

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
- Terminal asynchronous failure must either establish completion before physical release or retain resources in a faulted backend until teardown. Logical reclamation must not hide an unsuccessful physical release.
- Warp-cooperative GEMV, parallel model reductions, and native batched decode have same-artifact parity evidence on the RTX 4090. The float scalar path remains the explicit oracle.
- The measured serving baseline is approximately 20.2 tok/s at concurrency 1 and 31.7 aggregate tok/s at concurrency 8. These are historical measurements, not a performance claim for every later revision or device. [Execution history](../benchmarks/execution-history.md) records the qualification and measured progression.
- Experimental Q8_1 activation packing and standalone Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 integer-dot GEMV have focused CUDA layout, arithmetic, error-bound, and rejection tests plus kernel-level A/B wins over float warp at representative shapes. A batched Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 integer-dot template with per-member qualification and batch-8 wins is landed; batch-1 and batched-executor full-model integration are landed with token-identical greedy replay and a measured 1.23-1.33x serving throughput win at concurrency 2-8. The int-dot mode stays opt-in; the divergence gate (six diverse prompts, 256- and 1024-token matched greedy generations, all token-identical to warp) found no divergence, but a default flip would want broader sampling. Packed loads and integer dots are the next kernel hypothesis; they are not automatically qualified by the baseline results.
- The recurrent GDN state update runs as one batched kernel launch across all batch members; prefill and single-row decode keep the per-member kernel. The fused-sweep pass proved the kernel latency-bound (per-member launches cost 39.2 us each at 48 blocks, ~4% occupancy; halving traffic changed nothing), and batching cut its GPU-time share from 8.9% to 1.5% at concurrency 8 for a further ~1.10x serving throughput at every concurrency (c8 45.82 -> 50.60 tok/s int-dot). A three-member host-parity test covers the batched kernel including its unused-pointer-slot padding.

Next gates:

- Preserve the verified ownership contract while adding new execution variants. Admission errors return state, commitment validates before mutation, and failed reclamation retains a retry owner.
- CUDA lanes share prepared kernels and validated bindings; unsupported batch sizes use the per-row path. Driver faults prohibit new submissions, with uncertain resources retained until teardown.
- Real asynchronous CUDA cancellation and deferred release are verified, including a nine-row fallback followed by eight-row peer progress, and cancellation of a member inside a full eight-row batched-lane submission (peer tokens preserved, pinned-slot pool serviceable, registry drained). Injected host faults cover malformed completion, commitment, and release errors; deliberate destructive GPU fault injection is not part of this evidence.

Current deterministic decode-first policy can theoretically starve prefill if decode work continuously consumes the entire work budget. Treat bounded fairness as a measured scheduler-policy issue; add the smallest deterministic mechanism only if real workloads require it.

### 4D — state paging and exact reuse

A defensive validity guard is already present: fresh CUDA physical-state materialization at a nonzero logical prefix is rejected rather than allocating zero history and pretending it represents that prefix.

Remaining work:

- Implement real nonzero-prefix physical-state restoration/materialization.
- Add block/paged KV allocation where it improves real workloads.
- Preserve recurrent/other required state at reusable prefix boundaries.
- Add exact prefix reuse only when the complete model-required state can be reconstructed correctly.
- For Qwen3.8, reuse/restore must include both full-attention KV and recurrent/Gated-DeltaNet matrix plus convolution history at the same semantic prefix.
- Treat transfer, restore, reconstruction, and resident reuse as explicit materialization mechanisms rather than manufacturing a logical prefix from empty storage.
- Make allocation/reuse/transfer/materialization costs visible to scheduling.
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
- Allow phase-specific empirical costs and resource choices without making the scheduler architecture-specific.
- Versioned live policy with safe application boundaries and rollback.
- Transparent preparation cache for packed weights, compiled/JIT kernels, graphs, profiles, and warmup products.
- Exact artifact identity/invalidation and safe fallback; no surprise first-request compilation stall on a path reported ready.

The common path should first improve by removing host work, allocations, copies, and synchronization before relying on complex tuning.

## Phase 6 — speculation

- Use Qwen3.8 native MTP as the first speculation path.
- Treat propose/verify/state/cost behavior as a provider/runtime capability, not a Qwen-specific scheduler API.
- Add acceptance/cost telemetry and adaptive policy only after ordinary decode is competitive and stable.
- Qualify speculation jointly with graph mode, quantization, physical state representation, and hardware before automatic selection.

## Phase 7 — next model architectures and broader hardware coverage

Prioritize architectures and devices that force useful generalization rather than a long checklist of similar dense decoders.

### Next major native architecture pressure test — Qwen3.8-Flash-Next

After the current Qwen3.8-27B foundation, CUDA Rust migration, and state-boundary hardening are stable enough, make Qwen3.8-Flash-Next the next serious native model-family target.

It is a useful pressure test because it combines recurrent state, sparse/global attention, ultra-sparse MoE, model-native speculation, long context, multimodality, and a large host-resident N-gram lookup memory in one open model family. Its architecture is also explicitly presented as a preview of the direction toward Qwen4.

Implementation order should be driven by the real model rather than a speculative generic framework:

1. model metadata/provider description;
2. classify QSA structures into persistent semantic request state versus derived/backend scratch;
3. map the N-gram lookup memory as model-owned residency with an asynchronous host-prefetch path;
4. adapt/validate existing recurrent/GDN state semantics;
5. implement and qualify MoE execution in backend/runtime layers rather than teaching the common request scheduler expert identities;
6. add MTP speculation only after ordinary execution is correct;
7. add multimodal stages after the text path is stable unless a concrete artifact requires them earlier;
8. qualify long-context/state-memory behavior;
9. add performance variants only after exact correctness/reference evidence.

Primary architecture reference: <https://qwen.ai/blog?id=qwen3.8-flash-next>

### DeepSeek-V4.1-Flash — architecture compatibility spike before full support

Use DeepSeek-V4.1-Flash as a strong boundary stress test before committing to full native support. Its deployment scale makes it a less practical immediate local target than Qwen3.8-Flash-Next, but its architecture tests whether Ribn's semantic/physical boundaries remain coherent under very different choices.

The design spike should answer:

- Can causal-encoder/decoder prefill and decode be expressed without turning model structure into global request phases?
- Can cross-layer shared KV/index state fit semantic continuation requirements while physical sharing/layout remains backend-owned?
- Can FP4/block-scaled state storage be qualified without putting physical encoding in semantic state identity?
- Can bounded replay be represented as explicit state materialization rather than fake residency?
- Does Engram-style lookup memory map naturally to model-owned host residency?
- Does model-native DSpark speculation fit the existing speculation capability boundary?

If those answers require major core surgery, fix the boundary before broader Qwen4-like support makes the assumptions harder to remove. Full 552B-class model support should follow only when hardware/resources and user value justify it.

Primary references: <https://www.deepseek.com/en/news/deepseek-v4-1-flash/> and <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>

### Other architecture references

Use Kimi K3 and MiniMax M3 as independent evidence that the runtime should not assume one sparse/linear-attention mechanism. They are architecture references, not immediate implementation targets. Do not add KDA/MSA-specific core abstractions until Ribn actually targets those models.

References: <https://www.kimi.com/en/blog/kimi-k3> and <https://www.minimax.io/blog/minimax-m3>

### Benchmark gate for a new model family

Supporting a model is not itself the strategic win. For each serious new family, compare equivalent artifacts/quality/settings against the incumbents that can make a meaningful comparison, including vLLM, SGLang, llama.cpp where artifact/quantization semantics align, and TensorRT-LLM where practical.

Measure at minimum:

- time to first token;
- inter-token latency / TPOT;
- aggregate throughput across concurrency;
- accelerator and host memory consumption;
- startup/readiness time;
- long-context behavior;
- prefix/state reuse when implemented;
- model-owned host-prefetch overhead when applicable;
- performance per accelerator;
- token parity or model-appropriate numerical acceptance gates.

The useful strategic outcome is not merely broad model coverage; it is evidence that Ribn is unusually good at serving heterogeneous, stateful, sparse, or host-memory-assisted architectures without compromising correctness or runtime ownership semantics.

### Broader hardware

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

Engine remains a standalone inference runtime; fleet allocation belongs outside the project. Archon or another fleet manager may consume Ribn's distributed execution/resource options later, but it must not become a required dependency.

## Research track

Research can proceed alongside implementation but does not block the roadmap unless evidence shows a required boundary is wrong.

Potential directions include joint optimization of scheduling, execution variants, state placement/materialization, speculation, and topology; broader persistent/mega-kernel execution; state compression/reuse techniques; and new hardware-aware compilation strategies.

Do not promote a research idea into the common runtime merely because a frontier model demonstrates one mechanism. First classify whether the object is model-owned/static, request-owned persistent semantic state, reconstructable state, or submission-local backend scratch; then generalize only where concrete implementation pressure justifies it.

No novelty claim is required for the project to be useful. Near-term success is strong execution across performance, latency, memory efficiency, startup, portability, correctness, observability, configuration, and usability.
