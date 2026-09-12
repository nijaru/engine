# Inference engine research agenda

Date: 2026-09-12
Status: open design questions to resolve before stabilizing top-level contracts

This agenda exists to keep Ribn from converting promising ideas into permanent
architecture before they are compared against real models and workloads.

## 1. Model compatibility / day-zero bring-up

Native optimized Rust implementations should remain the production target, but a
new architecture should not require writing every optimized kernel before we can
validate config, processing, weights and numerics.

Compare at least these approaches:

1. optional Hugging Face Transformers/PyTorch reference bridge;
2. imported/exported graph representation where operator coverage is adequate;
3. Candle as a Rust-native reference/operator source;
4. Burn/CubeCL as a Rust-native portable operator/backend source;
5. direct small reference implementations for architectures where the above do
   not preserve the model faithfully.

Measure model coverage, remote-code/custom-model compatibility, multimodal
processors, numerical fidelity, operator gaps, startup cost, dependency footprint,
hardware support, and how difficult it is to replace reference ops with Ribn-native
optimized kernels.

Do not choose a fallback solely because it is written in Rust or because it has
broad model examples. A reference backend and the production serving backend have
different requirements.

## 2. Semantic operation layer versus direct model code

Research whether Ribn benefits from a small semantic operation layer inspired by
vLLM IR: model code names an operation with explicit semantics, while backend
implementations register support predicates and optimized kernels.

Potential benefits:

- model code can reuse the same semantic op across CUDA/ROCm/Metal/CPU;
- native/reference implementations become correctness oracles;
- kernel selection and fallback are observable and testable;
- fusion/compilation passes can reason about semantics rather than individual
  kernel symbols;
- backend-specific kernels remain specialized.

Risks:

- accidentally building a general ML compiler;
- dispatch or shape machinery leaking into the hot path;
- an IR too low-level to help model integration or too high-level to express new
  architectures;
- duplicating what a chosen compilation/backend substrate already provides.

Prototype only a few heavily reused operations (for example RMSNorm, quantized
linear, rotary/position transforms and one attention primitive) before deciding.
Static or preparation-time dispatch should be preferred over per-token dynamic
lookup.

## 3. Realtime / full-duplex sessions

Finite request/response is insufficient for current native realtime models. A
session may continuously accept time-aligned audio, video and text while producing
text/audio concurrently without a turn boundary.

Pressure-test a high-level `Session` concept distinct from one inference `Request`:

- stable session identity and model revision;
- multiple ordered input streams with timestamps/sequence numbers;
- partial input commits and backpressure;
- simultaneous output streams;
- interruption, half-close, cancellation and timeout semantics;
- conversation/model state lifetime across many internal stage operations;
- media jitter/buffering policy kept above model kernels;
- compatibility with WebSocket/realtime protocols without making WebSocket the
  internal API.

Do not force this into `TokenRequest` or treat every media chunk as a new model
request if the model itself is stateful across the session.

## 4. Cache/state identity and transfer

Dynamo, Mooncake, NIXL, LMCache and modern disaggregated engines make reusable
model state a distributed systems concern. Ribn should not own cluster routing, but
its state/resource interfaces should eventually permit external routing/storage and
direct transfer.

Research a versioned state identity/manifest that can describe:

- exact model/checkpoint revision and processor/input identity where relevant;
- semantic prefix or other continuation boundary;
- state component types and representation/layout versions;
- tensor/expert/pipeline parallel layout;
- precision/compression/quantization;
- owning device/rank and transfer compatibility;
- committed versus speculative state;
- content hashes/cache keys where appropriate.

Expose cache/resource events and transfer/export/import hooks only after local
resource semantics are correct. Keep transfer protocols (NIXL, Mooncake, RDMA,
NVLink, host/file stores) outside the AR scheduler.

## 5. Admission, QoS and overload

A production inference engine must be able to reject work before unbounded queues
destroy latency. Research admission based on real resource/cost estimates rather
than request count alone:

- queued prompt/encoder compute budget;
- continuation/cache memory pressure;
- expected decode or media-generation work;
- request deadline/priority;
- TTFT/ITL/E2EL/RTF service objectives depending on modality;
- preemption/recompute cost and cache locality.

Use HTTP 429/503 or protocol-appropriate errors at serving boundaries, while the
engine exposes structured overload/resource errors rather than strings.

## 6. Structured errors, diagnostics and capabilities

Replace string-only execution failures before public APIs stabilize. Errors should
separate unsupported model/input/config, invalid request, OOM/capacity, transient
resource pressure, device/kernel failure, cancelled/deadline, transport/stage
failure and poisoned/unsafe-to-continue execution.

Capabilities should be queryable from a loaded model and from the server: supported
operations/modalities, sampling/structured-output features, context/media limits,
quantization/backend variants, parallelism and known restrictions. Capabilities are
for inspection/validation, not a mandatory task-selection UI.

## 7. Hot weights, adapters and post-training inference

Modern engines are used inside RL/post-training loops as well as static serving.
Research lifecycle support for:

- LoRA/multi-LoRA adapter load/evict/select;
- sleep/wake/offload for colocated training and inference;
- atomic or versioned weight refresh;
- RDMA/P2P weight transfer from trainers;
- rollout request batching and reproducible sampling;
- hidden-state/logprob returns without routing them through text serialization;
- model-version pinning for in-flight requests.

Weight mutation must not invalidate cache/state silently. Model/version identity
needs to participate in continuation/cache compatibility.

## 8. Multi-model residency and routing boundary

A single process may eventually host multiple loaded models/adapters. Determine
what belongs inside Ribn versus an external router:

- model loading/unloading and residency budgets belong naturally with the engine;
- request routing among replicas based on cluster load/cache locality belongs more
  naturally to systems such as Dynamo/Archon;
- local selection among multiple resident models/adapters can remain an application
  or server concern.

Avoid hard-coding a fleet control plane into the model runtime.

## 9. Hardware abstraction pressure test

Do not call the backend boundary general until it survives another materially
different device stack. Compare NVIDIA CUDA with Apple Metal and/or AMD ROCm on:

- allocation/stream/queue/event ownership;
- graph/command-buffer capture semantics;
- async copies and unified memory;
- kernel specialization and compilation;
- collectives/distributed capabilities;
- quantization formats;
- host/device metadata movement.

Shared interfaces should express semantic requirements and lifetime/capability
information, not CUDA object shapes.

## 10. Benchmark and compatibility matrix

Maintain separate benchmarks for:

- model/kernel microbenchmarks;
- single-request local execution;
- offline batching;
- online continuous batching under arrival-rate control;
- shared-prefix/cache workloads;
- long context;
- multimodal encoder-heavy workloads;
- diffusion/media generation;
- realtime/full-duplex latency and RTF;
- multi-GPU/disaggregated execution.

Report modality-appropriate metrics: TTFT/ITL/TPOT and token throughput for AR,
E2EL/images or video seconds per second for media generation, RTF/TTFP for audio,
and latency/throughput for embeddings/reranking.

Protocol compatibility needs executable client tests, not endpoint-name matching.
Track OpenAI Responses/chat/completions/embeddings/audio/images where implemented,
Anthropic Messages/realtime where implemented, and any native Ribn extensions.

## Research references worth following

- vLLM V1 scheduler and Model Runner V2
- vLLM IR and vLLM-Omni
- SGLang / SGLang-Omni Unified Radix Cache and full-duplex roadmap
- TensorRT-LLM PyExecutor / ResourceManager
- MAX model pipeline and accelerator/compiler work
- mistral.rs Rust SDK/engine architecture
- llama.cpp / libmtmd
- MLC LLM compilation and edge deployment
- NVIDIA Dynamo / NIXL
- Mooncake / LMCache distributed state and transfer
- FlashInfer serving kernels and metadata/workspace patterns
- Candle and Burn/CubeCL as possible Rust ecosystem components

References are pressure tests and implementation sources, not architecture to copy
wholesale.
