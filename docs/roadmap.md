# Roadmap

This is an ordered engineering plan, not release dates. The current Qwen/4090
path is a qualification workload. The project target is a general inference
engine; architectural changes are expected when pressure tests expose a better
design.

[Inference engine design](inference-engine-design.md) is the target architecture.

## Current status

| Area | Current evidence | Main gap |
| --- | --- | --- |
| Qwen GGUF/CUDA AR | Legacy same-artifact references, new host lifecycle tests, CUDA-feature compilation | New-path GPU qualification; fixed-shape/model assumptions |
| AR runtime | Explicit ownership/cancellation, bounded per-request output, chunking, multi-token completion | Static full-sequence resources, separate prefill/decode queues, one in-flight batch |
| Text facade | Raw/chat/token inputs, tokenizer/template reuse, streaming/offline batch | Hardwired Qwen GGUF/CUDA loading; mutable single-caller handle |
| Model/artifact separation | QwenConfig independent of GGUF mapping | No general model registry, HF/safetensors/processor path |
| Multimodal/non-AR | Design research only | No general processor payloads, pooling/encoder runtime, iterative/diffusion runtime or orchestrator |
| CUDA Rust | Resource/toolchain gate complete | Representative quantized/recurrent kernels and full execution integration |
| Other hardware/distributed | Design only | No second backend or distributed execution path |

## 0. Reset the top-level architecture before deeper model-specific work

Treat `crates/runtime` as the current AR runtime rather than Ribn's universal
execution contract. Preserve its proven ownership behavior while introducing the
general boundaries in `docs/inference-engine-design.md`.

This stage should establish, with small executable tests rather than only traits:

- a loaded-model/model-package boundary that is not Qwen/GGUF/CUDA-specific;
- model capability/operation introspection without a user-visible task-default
  workflow;
- a shallow request orchestrator capable of one or more logical stages while the
  one-stage fast path remains direct;
- typed application inputs/outputs sufficient for text plus media and non-token
  results without putting raw media in the AR scheduler;
- first-class Hugging Face config/safetensors/tokenizer/processor loading alongside
  GGUF;
- a concurrent/cloneable Rust `Model` handle or equivalent whose driver owns
  mutable runtime state;
- clear separation between public/protocol requests and resolved model/runtime
  requests.

Do not stabilize crate/trait names before pressure tests. Reusing or moving current
code is preferred to adding a parallel third runtime hierarchy.

## 1. Pressure-test the architecture with different execution classes

Do this early enough that changing shared contracts is cheap. Full optimized model
support is not necessary for every test; small/reference-backed implementations can
expose a wrong boundary.

| Path | What it validates |
| --- | --- |
| Current Qwen hybrid AR | KV + recurrent state, quantization, chunked generation, cancellation |
| Dense decoder-only | New AR architecture without Qwen-specific changes |
| Encoder/pooling | Embeddings/scoring without fake AR sequences |
| Encoder-decoder ASR | Encoder output feeding decoding/cross-attention; audio input |
| VLM | Model-specific image/video processor + encoder + AR output |
| Diffusion image/video | Non-token iterative scheduling and media output |
| Non-AR text if practical | Text output does not imply autoregressive execution |

Select small models/configurations when possible so design validation does not turn
into months of kernel work.

## 2. Keep Qwen GPU qualification as the regression oracle

Run exact-artifact GPU replay and lifecycle tests through the new AR path while
architecture work proceeds. Cover concurrency 1/2/8/9, long outputs, mixed prompt
lengths, final prefill tails, cancellation in prefill/decode, stop/output limits,
repeated admission, constrained memory and fallback batch shapes.

Compare old/new TTFT, inter-token latency, throughput, CPU scheduling time,
allocations, host/device memory and preparation. Qwen remains a valuable hard test
because its hybrid continuation state prevents a KV-only design from looking more
general than it is.

## 3. Build dynamic continuation resources for the AR runtime

Replace full maximum-context reservation with model-appropriate dynamic resources.
The scheduler and resource manager should cooperate around capacity, prefix
matching/reuse, per-step preparation/update, eviction, preemption and release.

For attention state, evaluate paged allocation with compact page tables. For
recurrent/SSM state, evaluate checkpoint/copy-on-write/materialization strategies.
A reusable semantic prefix is valid only when every required component agrees on
the boundary.

Use SGLang's component-based Full/SWA/Mamba cache and current LMDeploy recurrent
checkpoint controls as reference designs, not mandatory implementations. Add
capacity pressure and cache/resource telemetry before choosing eviction or host
spill policy.

## 4. Redesign AR scheduling around the real resource model

Once dynamic resource costs exist, compare the current decode-priority + prefill
fairness queues with a unified scheduled-token budget similar in spirit to vLLM V1.
The scheduler should be able to express chunked prefill, cached progress,
speculative multi-token work, encoder-conditioned prompts and preemption without
special-case queue proliferation.

Measure FCFS/priority policy, token budgets, prefix locality, preemption/recompute
and admission watermarks against short/long/shared-prefix workloads. Preserve
prefill/decode as execution information when kernels need it; do not require those
to be top-level queue identities.

## 5. Make the host/device path async-first

Adopt the strongest relevant ideas from vLLM Model Runner V2 and FlashInfer:

- stable request rows for the active lifetime;
- persistent device metadata and staged incremental writes;
- packed/ragged batch descriptors and page tables;
- reusable caller/runtime-owned workspace and completion buffers;
- scheduler work for step N+1 overlapped with GPU work for step N;
- no normal-path device synchronization or metadata copy-back;
- GPU-side input metadata preparation/sampling where it reduces measured CPU
  overhead;
- explicit graph lifecycle and shape/compatibility checks;
- mixed prefill/decode kernels when measurements beat separated execution.

Benchmark allocation count, host scheduling, metadata bytes/copies, launch gaps and
GPU utilization independently from model FLOPs.

## 6. Continue CUDA Rust migration through the redesigned AR path

Prove representative Q8_1 packing -> quantized integer-dot projection and batched
GDN/recurrent state updates with numerical references, tails, repeated updates,
generated-code inspection and matched timings.

Integrate proven kernels through the current backend/resource ownership, not another
request runtime. Complete operation families incrementally, retaining vendor
libraries or existing kernels when they remain the best implementation.

Kernel-language migration and top-level runtime migration are separate proof gates.

## 7. Implement general model loading and model-support workflow

Make Hugging Face repository IDs/local directories, config JSON, safetensors,
tokenizer/chat-template and processor metadata first-class. GGUF remains supported
for quantized/local use.

Define the model-package/architecture registry from actual integrations: config,
weight adapters, processor, supported operations, logical stage topology,
continuation semantics and backend variants. New checkpoints of an existing
architecture should not require new scheduler/server code.

Research a compatibility/reference backend for day-zero bring-up. Compare an
optional Transformers/PyTorch bridge, portable graph import and Rust-framework
reuse on coverage, fidelity, operator support and maintenance. Do not make the
reference path the production hot path by accident.

## 8. Build the normal application and serving surfaces

The public Rust handle should be async/concurrent and cloneable while one driver
owns mutable model/runtime state. Keep a lower-level direct API for embedding and
specialized control.

Implement public operations as their actual semantics appear: generation/chat,
embedding/scoring/classification, transcription/translation, speech or media
output, etc. Do not force these through one generic task selector.

Then add `ribn serve` with a tested compatibility subset. Cover streaming,
structured errors, finish reasons, usage/logprobs, disconnect cancellation,
request IDs, bounded admission, health/readiness and metrics. Add OpenAI/Anthropic
routes where relevant and native extensions only when they expose useful engine
capability.

Python in-process bindings are an important follow-on for evaluation, RL/post-
training and the existing ML ecosystem; HTTP is not a sufficient replacement.

## 9. Multimodal and media pipelines

Implement media content parts and model processors with explicit ownership and
security limits. A VLM path should keep processor placeholder/position semantics
model-specific while sharing encoder scheduling/cache and AR serving machinery.

For omni/diffusion models, compose logical stages only where the model has genuinely
different execution loops. Allow co-located stages to pass device-resident payloads
without serialization. Add cross-stage cancellation/output ordering before
supporting distributed placement.

Add encoder/media-result caching only with identity that includes source contents,
processor/config/model revision and representation compatibility.

## 10. Second hardware backend and distributed inference

Use a materially different backend (Metal/Apple or AMD when the implementation is
ready) to test the resource/device boundary. Shared request/model code should not
require CUDA streams, memory layouts or graph semantics.

Ribn may then add inference-local tensor/expert/pipeline/data/context parallelism,
collectives, prefill/decode disaggregation and stage/model-state transfer. Keep
logical model topology separate from deployment topology. External systems allocate
and place resources; Ribn owns execution within those resources.

## 11. Cut over and delete transitional architecture

When the new AR path passes GPU correctness/lifecycle/performance gates and the
general model boundary has survived pressure tests, remove the legacy serving
runtime and Qwen's compatibility translation. Relocate surviving helpers to their
real model/backend/resource owners.

Do not maintain old/new feature matrices indefinitely. Historical commits and
reference fixtures preserve the oracle without shipping two architectures.

## Strategic benchmark

Ribn succeeds if it can add modern models without architectural surgery and deliver
strong latency, throughput, memory efficiency, correctness, reliability and
integration ergonomics across local, embedded and serving workloads.

A clean abstraction, Rust implementation or long feature list is not itself the
result. Rewrites are appropriate when evidence shows the current design is the
wrong substrate; the project is early enough that avoiding a necessary redesign is
more expensive than doing it now.
