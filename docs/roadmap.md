# Roadmap

This is an ordered engineering plan, not release dates. The current Qwen/4090
path is a qualification workload. The project target is a general inference
engine; architectural changes are expected when pressure tests expose a better
design.

[Inference engine design](inference-engine-design.md) is the target architecture.
[Shared execution foundation](execution-foundation.md) records the provisional
boundary beneath inference-specific runtimes. [Resource, submission and snapshot
protocol](resource-protocol.md) records the reservation, handoff and snapshot
contracts those runtimes need next. [Pipeline composition](pipeline-composition.md)
records the distinction between genuinely staged execution and tightly coupled
cross-component scheduling.

## Current status

| Area | Current evidence | Main gap |
| --- | --- | --- |
| Shared execution foundation | Dependency-free parameter/version/materialization metadata; node/device/link topology; same logical model placed locally or across nodes; preparation-time semantic-op dispatch experiment | Physical storage/device primitives, broader operator/backend evidence, second hardware backend |
| Qwen GGUF/CUDA AR | Legacy same-artifact references, new host lifecycle tests, CUDA-feature compilation; same-sequence multi-token prefill is hardware-qualified at `4c22e11` (single-chunk, three-chunk, serving-seam, and end-to-end runtime gates; 1.75x prefill at the 8-member cap and 1.75x serving TTFT at a 257-token prompt with identical tokens and no decode change) and is selected by default through `QwenLoadOptions` | True chunked GDN/full-attention prefill, since the lane batches projections and feed-forward work per position while recurrence still advances token by token; long-prompt numerical fidelity, where a 257-token prompt flips one llama.cpp near-tie of 0.2242 nats with chunking both on and off; fixed-shape/model assumptions |
| AR runtime | Explicit ownership/cancellation, bounded per-request output, chunking, multi-token completion; stable `RequestId` is passed into executor admission separately from `SequenceId` | Physical demand is invisible to admission, static full-sequence resources, separate prefill/decode queues, one in-flight batch, production resource-planner cooperation ([resource protocol](resource-protocol.md)) |
| Non-AR runtime validation | Generic batch runtime bounding waiting work and retained results separately, with ready/blocked/rejected admission and a rejection path that survives an exhausted byte budget; parameter-version pinning; actual BERT encoder semantics; attention masks and padded-versus-ragged batch-cost tests | Async/cancellation/resource admission, shared-pool accounting, device execution and optimized kernels |
| Text facade | Raw/chat/token inputs, tokenizer/template reuse, streaming/offline batch | Hardwired Qwen GGUF/CUDA loading; mutable single-caller handle |
| Model/artifact separation | `QwenConfig` independent of GGUF; SafeTensors adapter retaining tensor metadata and offsets once for indexed lookups; local HF-style config + unsharded/sharded weight-package resolver including symlinked cache snapshots; `LocalWeightSet` with a configurable residency bound; Qwen and BERT integrations keep model meaning above artifact parsing | Remote HF repository/revision resolution, tokenizer/processor package integration, architecture resolution when a second production model justifies it, backend materialization path |
| Cross-runtime composition | Sequential batch-encoder -> AR handoff passes prepared state in-process; stable request identity survives out-of-order handoff; cancellation ownership is validated before and after AR admission | Genuine encoder-decoder model, cross-attention/device-state lifetime, async failure propagation and version compatibility |
| Multimodal/iterative | VLM pressure test models prompt-positioned encoder items with independent encoder-compute and encoder-cache pressure; staged-vs-coupled distinction is validated | Typed media processor path, real VLM integration and production coupled scheduler/resource seam; iterative/diffusion runtime; realtime session path |
| CUDA Rust | Resource/toolchain gate complete | Representative quantized/recurrent kernels and full execution integration |
| Other hardware/distributed | Resource-topology/placement representation only | No second backend, collectives, remote execution or state transfer |

## Milestone order and exit evidence

The numbered sections below describe the work in each area. They are not the order
to do it in. Adding another model-class pressure test is cheap and feels productive,
but every one of them inherits the same unresolved lifecycle: unbounded retention,
reservation that cannot fail before execution, and state whose ownership only covers
allocation. Resolve the lifecycle first and the later counterexamples become small.

| Milestone | Required exit evidence |
| --- | --- |
| 1. Bounded, concurrent execution lifecycle | A minimal cloneable model handle with explicit request ownership; bounded retained output; admission that distinguishes ready, temporarily blocked, and rejected; an owning snapshot. Exercised with multiple callers and a stalled consumer. |
| 2. One real asynchronous non-AR integration | Encoder work against real device resources producing owned results, surviving cancellation and handoff failure without premature reclamation. |
| 3. Dynamic hybrid AR resources | KV and recurrent state cooperating with reservations, cache reuse, eviction and preemption. Non-speculative correctness first, then qualified speculative reconciliation. |
| 4. Real composition counterexamples | A genuine VLM processor/model path and a small iterative non-AR path, changing shared contracts only where those integrations demonstrate the need. |
| 5. Qualification and broader exposure | Matched numerical and workload benchmarks, a tested serving subset, and a materially different backend before calling the shared device boundary general. |

Milestone 1 has partially started: the non-AR runtime now bounds retained results and
separates blocked from rejected admission, and the resource protocol records the
reservation, handoff and snapshot contracts the rest of it needs. The public handle,
explicit request ownership, and snapshot ownership are not implemented.

Two tracks stay independent of this order instead of becoming prerequisites:

- CUDA kernel-language migration keeps its own proof track ([cuda rust migration](cuda-rust-migration.md));
  existing kernels and vendor libraries stay the qualified implementation while
  runtime correctness is proven.
- Distributed placement types stay provisional until real sharding, replication or
  cross-device execution constrains them. Metadata accepting several device IDs is
  not yet a distributed execution abstraction.

## 0. Reset and validate the top-level architecture before deeper model-specific work

Treat `crates/runtime` as the current AR runtime rather than Ribn's universal
execution contract. Preserve its proven ownership behavior while introducing the
general boundaries in `docs/inference-engine-design.md`.

Executable validation scaffolds now establish several useful boundaries:

- `ribn-foundation` separates logical parameter/version identity from physical
  materialization metadata and keeps request/token/KV/autograd policy out of the
  dependency base;
- resource topology describes nodes/devices/links independently of model semantics;
- a provisional execution plan can place the same logical model/version on one
  device or across nodes without changing model identity;
- a test-only semantic-operator experiment selects specialized/reference
  implementations during preparation rather than walking a registry in the hot
  execution path;
- `ribn-batch` demonstrates non-AR execution with executor-defined inputs/outputs,
  parameter-version pinning, and executor-informed FIFO batch sizing without fake
  token/prefix/KV semantics;
- `ribn-safetensors` validates artifact bytes and exposes borrowed format-level
  tensor views without assigning model semantics or allocating execution tensors;
- `ribn-hf` resolves a local HF-style `config.json` plus single or sharded
  SafeTensors weights without choosing model architecture, runtime or backend;
- a test-only BERT architecture loads that HF/SafeTensors package and executes real
  encoder structure—embeddings, multi-head self-attention, residual/LayerNorm, FFN
  and pooler—through `ribn-batch`, while sequence length remains an executor-owned
  batching constraint;
- that BERT loading path also exposed a concrete model-loader requirement: one
  SafeTensors shard should be opened once and serve many parameter views rather than
  being reread independently for every tensor;
- the AR executor admission seam receives stable `RequestId` separately from
  internal `SequenceId`; sequential encoder->decoder tests prove prepared state can
  remain in-process, correlate correctly independent of handoff order, and transfer
  cleanup ownership at admission;
- a VLM scheduler pressure test proves prompt-positioned encoder items can constrain
  AR prefill and that encoder compute pressure and encoder-cache pressure are
  independent scheduling facts. This is evidence for scheduler/resource-planner
  cooperation, not for a universal resource-cost vector or raw media in the AR
  request.

These tests prove that the separation is implementable, **not** that the exact type
shapes are finished. `StageId`, `RuntimeClass`, materialized storage IDs, current
topology fields, `BatchExecutor::select_batch`, the request-admission seam, and the
VLM test's dependency representation remain pressure-test interfaces.

Remaining work in this architecture-validation stage:

- pressure-test real device resource costs, asynchronous execution, cancellation
  and failure behavior on the encoder path; masks plus padded/ragged host execution
  already fit executor-owned batch selection without a universal cost unit;
- establish the general loaded-model/model-package and architecture-resolution
  boundary when another production model makes it useful. Qwen is still the only
  production model path and BERT is a pressure-test integration, so a registry now
  would mostly formalize strings rather than remove real duplication. An ordinary
  central Rust enum/registry/factory remains acceptable when evidence justifies it;
- add tokenizer/processor/package metadata and model capability/operation
  introspection without a user-visible task-default workflow;
- integrate a genuine encoder-decoder model to validate cross-attention state,
  device-resident handoff, cancellation/failure propagation and version
  compatibility;
- integrate a real VLM/processor path and use it to turn the test-only
  prompt-position dependency model into the minimum production scheduler/resource
  seam actually required;
- introduce typed application inputs/outputs sufficient for text plus media and
  non-token results without putting raw media in the AR scheduler;
- build a concurrent/cloneable Rust `Model` handle or equivalent whose driver owns
  mutable runtime state;
- extend the semantic-op experiment to a few real operations/backend
  implementations and discard it if it turns into compiler machinery without
  practical benefit;
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

This is not a license to keep adding model classes ahead of the lifecycle. A new
pressure test is worth building when it exposes a shared-contract problem no existing
test can, or when its milestone in the table above is reached. Model classes that
would only inherit the same unbounded retention, unreservable submission, and
allocation-only ownership wait, because running them proves nothing new about the
boundary.

| Path | Current state | What it validates |
| --- | --- | --- |

The `Current state` column records which pressure test ran, not what it proved.
`docs/execution-foundation.md` holds the validated behavior and its limits, and is the
single owner of evidence; this table links to that rather than restating it.
| Current Qwen hybrid AR | Existing prototype | KV + recurrent state, quantization, chunked generation, cancellation |
| Dense decoder-only | Not started | New AR architecture without Qwen-specific changes |
| Encoder/pooling | Actual BERT architecture reference path passes through HF/SafeTensors + `ribn-batch`; sequence-length batch selection, attention masks, and padded-versus-ragged cost pass | Non-AR model semantics do not need AR contracts; next: real device resource admission and asynchronous execution |
| Sequential encoder-decoder | Cross-runtime state handoff and cancellation ownership pass in-process using stable AR `RequestId` | Next: genuine encoder-decoder model, cross-attention/device state, async failure/version lifetime |
| VLM | Prompt-positioned encoder-dependency pressure test passes with separate encoder compute/cache constraints | Next: actual processor/model integration and minimum production scheduler/resource interface |
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

The VLM pressure test already shows that a coupled generation path sometimes needs
per-item prompt-span readiness plus independent encoder-compute and encoder-cache
capacity. Do not turn those two demonstrated resources into an arbitrary generic
resource vector. Let an actual VLM integration determine whether the production
boundary is explicit dependency descriptors, a model/resource planner queried by
the scheduler, or another small cooperative interface.

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

Local HF-style package and SafeTensors artifact boundaries now exist as validation
code. Qwen and BERT provide two concrete model-family integrations against which to
pressure-test architecture resolution. Extend these layers deliberately rather than
making the artifact parser responsible for model semantics.

Repeated tensor access is now practical without moving model semantics into the
package layer: `LocalWeightSet` lazily opens each resolved SafeTensors shard once and
reuses it for many borrowed tensor views. Backend-specific prepared materialization
and streaming/loading policy remain future work.

Add Hugging Face repository IDs/revisions, tokenizer/chat-template and processor
metadata as first-class sources alongside local directories. GGUF remains supported
for quantized/local use.

Define the model-package/architecture resolver from actual integrations: config,
logical parameter identity, weight adapters/materializations, processor, supported
operations, logical stage topology, continuation semantics and backend variants.
New checkpoints of an existing architecture should not require new scheduler/server
code. A central enum or registry changing when genuinely new architectures are
added is acceptable; the failure mode to avoid is model-specific branching scattered
through unrelated runtime policy.

Research a compatibility/reference backend for day-zero bring-up. Compare an
optional Transformers/PyTorch bridge, portable graph import and Rust-framework
reuse on coverage, fidelity, operator support and maintenance. Do not make the
reference path the production hot path by accident.

Hot-weight/adaptor updates should prepare a new coherent parameter version and
commit it at a safe boundary. Do not silently retain caches, recurrent checkpoints,
encoder outputs, or compiled/captured execution variants across incompatible
versions.

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
model-specific while sharing useful encoder scheduling/cache and AR serving
machinery.

Do not assume every encoder is a separate top-level stage. Sequential
encoder-decoder models can use a shallow staged handoff, while VLM/omni models may
need a coupled runtime where encoder readiness/cache and AR prompt progress are
scheduled together. The synthetic VLM pressure test has validated that distinction;
an actual VLM must now determine the concrete production seam. See
`docs/pipeline-composition.md`.

For genuinely staged omni/diffusion models, compose logical stages where the model
has different execution loops. Allow co-located stages to pass device-resident
payloads without serialization. Add cross-stage cancellation/output ordering before
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
