# Roadmap

This is an ordered engineering plan, not release dates. The current Qwen/4090
path is a qualification workload. The project target is a general inference
engine; architectural changes are expected when pressure tests expose a better
design.

[Inference engine design](inference-engine-design.md) is the target architecture.
[Shared execution foundation](execution-foundation.md) records the provisional
boundary beneath inference-specific runtimes.

## Current status

| Area | Current evidence | Main gap |
| --- | --- | --- |
| Shared execution foundation | Dependency-free parameter/version/materialization metadata; node/device/link topology; same logical model placed locally or across nodes in tests | Real model loading, physical storage/device primitives, operator/backend experiment, second hardware backend |
| Qwen GGUF/CUDA AR | Legacy same-artifact references, new host lifecycle tests, CUDA-feature compilation | New-path GPU qualification; fixed-shape/model assumptions |
| AR runtime | Explicit ownership/cancellation, bounded per-request output, chunking, multi-token completion | Static full-sequence resources, separate prefill/decode queues, one in-flight batch |
| Non-AR runtime validation | Generic bounded batch runtime executes non-token inputs and pins queued work to a parameter version | Real encoder/pooling model, shape/memory-aware batching, async/cancellation/resource admission |
| Text facade | Raw/chat/token inputs, tokenizer/template reuse, streaming/offline batch | Hardwired Qwen GGUF/CUDA loading; mutable single-caller handle |
| Model/artifact separation | QwenConfig independent of GGUF mapping | No general model registry, HF/safetensors/processor path |
| Multimodal/iterative | Synthetic placement only | No general processor payloads, multistage orchestrator, iterative/diffusion runtime or realtime session path |
| CUDA Rust | Resource/toolchain gate complete | Representative quantized/recurrent kernels and full execution integration |
| Other hardware/distributed | Resource-topology/placement representation only | No second backend, collectives, remote execution or state transfer |

## 0. Reset and validate the top-level architecture before deeper model-specific work

Treat `crates/runtime` as the current AR runtime rather than Ribn's universal
execution contract. Preserve its proven ownership behavior while introducing the
general boundaries in `docs/inference-engine-design.md`.

The first executable validation scaffolds are now implemented:

- `ribn-foundation` separates logical parameter/version identity from physical
  materialization metadata and keeps request/token/KV/autograd policy out of the
  dependency base;
- resource topology describes nodes/devices/links independently of model semantics;
- a provisional execution plan can place the same logical model/version on one
  device or across nodes without changing model identity;
- `ribn-batch` demonstrates a non-AR runtime can batch executor-defined inputs and
  outputs without fake token/prefix/KV semantics;
- queued non-AR work is pinned to a coherent parameter version and rejects an
  uncoordinated executor-version change.

These tests prove the split is implementable, **not** that the exact type shapes are
finished. In particular, `StageId`, `RuntimeClass`, item-count batching, materialized
storage IDs, and the current topology fields remain pressure-test scaffolding.

Remaining work in this architecture-validation stage:

- run a real small encoder/pooling model through the non-AR path and let its shapes,
  memory and batching requirements change the provisional contracts;
- add a loaded-model/model-package boundary that is not Qwen/GGUF/CUDA-specific;
- establish model capability/operation introspection without a user-visible
  task-default workflow;
- pressure-test a shallow orchestrator with a genuine encoder-decoder or multimodal
  multi-stage flow while keeping the one-stage fast path direct;
- introduce typed application inputs/outputs sufficient for text plus media and
  non-token results without putting raw media in the AR scheduler;
- make Hugging Face config/safetensors/tokenizer/processor loading first-class
  alongside GGUF;
- build a concurrent/cloneable Rust `Model` handle or equivalent whose driver owns
  mutable runtime state;
- prototype a tiny semantic-op/backend-selection layer on a few operations and
  discard it if it turns into compiler machinery without practical benefit;
- pressure-test the shared device/resource boundary against a materially different
  backend before calling it general;
- keep the shared foundation reusable by a future trainer without adding autograd,
  gradient, optimizer or training-scheduler semantics to Ribn inference.

Do not stabilize crate/trait names before these pressure tests. Reusing or moving
current code is preferred to adding a parallel hierarchy.

## 1. Pressure-test the architecture with different execution classes

Do this early enough that changing shared contracts is cheap. Full optimized model
support is not necessary for every test; small/reference-backed implementations can
expose a wrong boundary.

| Path | Current state | What it validates |
| --- | --- | --- |
| Current Qwen hybrid AR | Existing prototype | KV + recurrent state, quantization, chunked generation, cancellation |
| Dense decoder-only | Not started | New AR architecture without Qwen-specific changes |
| Encoder/pooling | Generic runtime scaffold only; **next real-model pressure test** | Embeddings/scoring without fake AR sequences; real shape/memory batching |
| Encoder-decoder ASR | Not started | Encoder output feeding decoding/cross-attention; audio input |
| VLM | Not started | Model-specific image/video processor + encoder + AR output |
| Diffusion image/video | Not started | Non-token iterative scheduling and media output |
| Realtime/full-duplex | Design only | Long-lived session identity, concurrent media input/output, interruption/backpressure |
| Non-AR text if practical | Not started | Text output does not imply autoregressive execution |

Select small models/configurations when possible so design validation does not turn
into months of kernel work. A pressure-test implementation may be replaced after it
has exposed the boundary issues it was built to find.

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
the boundary. Add explicit retain/fork/copy-on-write/checkpoint/materialize semantics
where branching, speculative decoding, parallel candidates or RL rollouts require
them rather than assuming every continuation is linear.

Use component-based hybrid caches as reference designs, not mandatory
implementations. Add capacity pressure and cache/resource telemetry before choosing
eviction or host spill policy. Cache/state identity must include the model and
parameter/adaptor version that produced it.

## 4. Redesign AR scheduling around the real resource model

Once dynamic resource costs exist, compare the current decode-priority + prefill
fairness queues with a unified scheduled-token budget. The scheduler should be able
to express chunked prefill, cached progress, speculative multi-token work,
encoder-conditioned prompts and preemption without special-case queue proliferation.

Measure FCFS/priority policy, token budgets, prefix locality, preemption/recompute
and admission watermarks against short/long/shared-prefix workloads. Preserve
prefill/decode as execution information when kernels need it; do not require those
to be top-level queue identities.

## 5. Make the host/device path async-first

Target:

- stable request rows for the active lifetime;
- persistent device metadata and staged incremental writes;
- packed/ragged batch descriptors and page tables;
- reusable caller/runtime-owned workspace and completion buffers;
- scheduler work for step N+1 overlapped with device work for step N;
- no normal-path device synchronization or metadata copy-back;
- device-side input metadata preparation/sampling where it reduces measured CPU
  overhead;
- explicit graph lifecycle and shape/compatibility checks;
- mixed prefill/decode kernels when measurements beat separated execution.

Benchmark allocation count, host scheduling, metadata bytes/copies, launch gaps and
accelerator utilization independently from model FLOPs.

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
logical parameter identity, weight adapters/materializations, processor, supported
operations, logical stage topology, continuation semantics and backend variants.
New checkpoints of an existing architecture should not require new scheduler/server
code.

Research a compatibility/reference backend for day-zero bring-up. Compare an
optional Transformers/PyTorch bridge, portable graph import and Rust-framework
reuse on coverage, fidelity, operator support and maintenance. Do not make the
reference path the production hot path by accident.

Hot-weight/adaptor updates should prepare a new coherent parameter version and
commit it at a safe boundary. Do not silently retain caches, recurrent checkpoints,
or compiled/captured execution variants across incompatible versions.

## 8. Build the normal application and serving surfaces

The public Rust handle should be async/concurrent and cloneable while one driver
owns mutable model/runtime state. Keep a lower-level direct API for embedding and
specialized control.

Implement public operations as their actual semantics appear: generation/chat,
embedding/scoring/classification, transcription/translation, speech or media
output, etc. Do not force these through one generic task selector.

Then add `ribn serve` with a tested compatibility subset. Cover streaming,
structured errors, finish reasons, usage/logprobs, disconnect cancellation,
request IDs, bounded admission, health/readiness and metrics. Add expected protocol
routes where relevant and native extensions only when they expose useful engine
capability.

Python in-process bindings are an important follow-on for evaluation, RL/post-
training and the existing ML ecosystem; HTTP is not a sufficient replacement.

## 9. Multimodal, media and session pipelines

Implement media content parts and model processors with explicit ownership and
security limits. A VLM path should keep processor placeholder/position semantics
model-specific while sharing encoder scheduling/cache and AR serving machinery.

For omni/diffusion models, compose logical stages only where the model has genuinely
different execution loops. Allow co-located stages to pass device-resident payloads
without serialization. Add cross-stage cancellation/output ordering before
supporting distributed placement.

For realtime/full-duplex models, pressure-test a long-lived session lifecycle that
can accept ordered/timestamped audio/video/text streams while emitting multiple
output streams. Do not model every chunk as a new AR request when the model itself
holds state across the session.

Add encoder/media-result caching only with identity that includes source contents,
processor/config/model revision, parameter/adaptor version and representation
compatibility.

## 10. Second hardware backend and distributed inference

Use a materially different backend (Metal/Apple or AMD when the implementation is
ready) to test the resource/device boundary. Shared model/foundation code should
not require CUDA streams, memory layouts or graph semantics.

Ribn may then add inference-local tensor/expert/pipeline/data/context parallelism,
collectives, prefill/decode disaggregation and stage/model-state transfer. Keep
logical model topology separate from deployment topology. External systems allocate
and place resources; Ribn owns execution within those resources.

Distributed capability must be additive: a one-device plan should collapse to a
direct in-process runtime/device path without RPC, serialization, synthetic workers,
or cluster-control overhead.

## 11. Cut over and delete transitional architecture

When the new AR path passes GPU correctness/lifecycle/performance gates and the
general model/foundation boundary has survived pressure tests, remove the legacy
serving runtime and Qwen's compatibility translation. Relocate surviving helpers to
their real model/backend/resource owners.

Do not maintain old/new feature matrices indefinitely. Historical commits and
reference fixtures preserve the oracle without shipping two architectures.

## Strategic benchmark

Ribn succeeds if it can add modern models without architectural surgery and deliver
strong latency, throughput, memory efficiency, correctness, reliability and
integration ergonomics across local, embedded, serving and distributed workloads.

A clean abstraction, Rust implementation or long feature list is not itself the
result. Rewrites are appropriate when evidence shows the current design is the
wrong substrate; the project is early enough that avoiding a necessary redesign is
more expensive than doing it now.
