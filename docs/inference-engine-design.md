# Inference engine design

Status: accepted target, not a support claim. Public signatures and prepared-resource
representations remain unstable. [Resource protocol](resource-protocol.md) owns
execution ownership; [roadmap](roadmap.md) owns sequencing and unresolved decisions;
[architecture](architecture.md) describes current code.

## Scope and success criterion

Ribn targets a state-of-the-art, idiomatic Rust, **server-first** inference engine,
initially competitive with vLLM, SGLang and similar engines. Throughput within latency
objectives, memory efficiency, model-support velocity, hardware breadth and reliability
are joint goals. Correct ownership is an enabler, not a smaller correctness-only niche.

Qwen GGUF/CUDA and the development RTX 4090 are qualification workloads, not the
product boundary. Qualify selected workloads first, then expand coverage. Compare
pinned competitor configurations on matched hardware, artifacts, quality settings and
mixed workloads; report gaps and regressions. No current SOTA or broad-support claim
follows from this design.

Server-first prioritizes sustained concurrency, fairness, overload, cancellation and
bounded memory over local startup convenience. It does not require IPC or multiple
processes. CLI, Rust, Python and HTTP must reuse model behavior and execution ownership,
not implement separate schedulers.

All v0 APIs are unstable. Replace flawed contracts and remove superseded paths rather
than adding compatibility shims. Preserve released behavior only where an actual
commitment exists; preserve correctness evidence regardless.

## Four boundaries

1. **Artifacts and model definition:** validated configuration, processors, parameter
   mappings, operation semantics and supported capabilities; no serving lifecycle.
2. **Prepared execution:** owning backend-specific weights, storage, qualified kernels,
   completion dependencies and resource ownership.
3. **Execution policy:** specialized AR, batch/encoder and later iterative or training
   runtimes. Share mechanisms only when real consumers demonstrate reuse.
4. **Application access:** operation-specific owned requests/results, bounded submission,
   cancellation, diagnostics and frontend adapters.

```text
CLI / Rust / Python / HTTP
            |
operation-specific model handle + processor
            |
   +--------+----------------+----------------+
   |                         |                |
AR token runtime       batch / encoder    iterative runtime
continuation           pooling / scores    diffusion / other
   |                         |                |
   +------ model/backend execution -----------+
                         |
             storage / devices / kernels
```

A single-runtime model crosses one application boundary to one mutable execution owner.
Do not insert orchestrator/worker/executor/engine layers that forward identical calls.
A shallow orchestrator is justified only for genuine cross-runtime dependencies: routing,
cancellation, output ordering, placement and bounded handoff. It does not allocate KV
pages, schedule attention tokens or choose kernels.

`crates/runtime` (`ribn`) is the AR runtime, not the universal inference API. Text output
does not imply autoregressive execution. Encoders, speech encoder-decoder models,
multimodal/omni and diffusion/media models may need materially different loops.

## Application API and UX

Users load or serve a model and invoke a supported operation; internal runtime selection
is not a mandatory task-default ceremony.

- CLI: `ribn run <model>` and inspection/benchmark utilities. Add operation-specific
  commands only when semantics genuinely differ.
- Server: `ribn serve <model>` with a tested, documented protocol subset, health/readiness,
  metrics, bounded admission, disconnect cancellation and security limits.
- Rust/Python: cloneable in-process loaded handles with owned responses/streams and
  operation-specific methods such as generate/chat/embed/score/transcribe. Keep a direct,
  channel-free runtime surface for embedding and specialized scheduling.
- Use conventional operation-appropriate protocols: chat/Responses, embeddings, reranking,
  transcription, speech and image/video generation are not interchangeable payloads.

These are targets. There is no HTTP server or general model loader yet.

Keep one mutable execution owner. Blocking and async clients drive the same engine;
async submission must not block an executor with GPU polling or tokenization. An owned
stream can outlive submission, yields ordered deltas and one terminal outcome, and owns
cancellation interest. Dropping it abandons delivery without waiting for device completion.
Direct use does not require a worker thread. Thread-affine construction needs a real
backend contract before it is added.

Separate load/resource configuration, scheduler policy, per-request generation settings
and serving settings. Validate defaults and let explicit request values override supported
model defaults. Prefer ordinary structs/builders and typed errors to typestate machinery
or an `infer(Any) -> Any` interface. Unsupported operations fail explicitly.

Bound preprocessing, submitted inputs, output buffers and physical resources separately.
Offline batching admits a bounded window, yields ordered per-item results and does not
collect an arbitrary iterator first. A stalled consumer cannot block cancellation or
healthy peers. Caller-collected results are distinct from buffered application storage.
The [text contract](resource-protocol.md#text-application-facade) owns current details.

## Model and artifact ownership

A model family owns configuration semantics, parameter interpretation, processors,
special-token behavior, public capabilities, continuation semantics and backend execution.
Registration is centralized selection, not family branches scattered through lifecycle,
routing and protocols. A second real decoder and real VLM must prove this contribution
boundary before routine integration is advertised.

GGUF and SafeTensors expose format data, not model semantics. HF package resolution
owns source/revision/config/weight files; it does not choose execution policy. Remote HF
resolution, tokenizer/processor metadata and architecture registration remain future work.
GGUF is useful but must not define the whole loading architecture.

Queued inputs are owned or share explicitly immutable allocations. Raw media, fetching
policy and preprocessing stay above token scheduling. Actual operations should introduce
typed text/messages, images, audio, video and tensor interchange where needed; do not
encode every modality as fictional text tokens. Network APIs must not expose backend
CUDA/Metal tensor types. Structured/media outputs need their own representation and bounds.

Native optimized execution is the production target. Evaluate an optional reference path
for bring-up and numerical comparison: Transformers/PyTorch, exported graphs or suitable
Rust components are candidates, not decisions. Measure fidelity, model/operator coverage
and maintenance cost. Python must not become a mandatory production runtime for fallback
coverage. A registry alone cannot provide automatic day-zero model support.

## Continuation resources and prefix reuse

Full-reachable reservation is the safe current Qwen policy, not the serving destination.
The target is backend-owned block storage, aggregate pre-submit growth negotiation,
completion-safe settlement, valid hybrid reuse and a progress policy under pressure.
[Resource protocol](resource-protocol.md#ar-continuation-resources) owns those transitions.

- Declared component shape and maximum capacity are not reservations. Charge only concrete
  storage; growth is a reservation transaction, not an edit of the declaration.
- Separate a physical arena/slab's resident byte charge from occupancy of reusable block
  slots. Evicting a slot does not free the still-resident arena's bytes to sibling runtimes.
- Start with snapshot-scoped, content-keyed fixed blocks rather than a token-granular radix
  tree. Parent identity plus tokens identifies content, not proof of live allocations.
  Restore only where all required KV ancestors and recurrent/other components remain valid.
- A hybrid KV match without matching recurrent state is not a continuation. Checkpoints
  must be sparse, and the cache may decline reuse. Restore recurrent state into a private
  writable allocation; copying and temporary overlap cost real memory.
- Initial restoration must stop strictly before the prompt's final sampling token, then
  execute the suffix to obtain logits. A checkpoint after that token cannot simply consume
  it again in a recurrent model. Cached logits/output state is a separate, deferred policy.
- Eviction reclaims unreferenced cache occupancy. Preemption reclaims live continuation
  only after completion, retaining bounded replay history. Start with recomputation, not
  a host swap tier. Replay cannot duplicate delivery, usage or sampling.
- Before partial-envelope admission, prove progress: readiness alone cannot break a pool
  held by active sequences all waiting to grow. Survivor growth must take precedence over
  victim readmission; permanently infeasible work must reject locally.

The pinned hybrid has 149.6 MiB of recurrent state per sequence versus 16.8 MiB of KV at
257-token reach. That evidence argues against dense checkpoints; it does not show that
reuse improves short-request concurrency. Compare paging with and without reuse on a
justified shared-prefix geometry, including checkpoint residency/copy peaks and TTFT.

## Execution and performance direction

Keep physical layouts, storage, kernels and device submission backend-native. Idiomatic
Rust does not mean Rust-only kernels. Qualified vendor/external kernels are appropriate;
do not build a compiler, tensor IR, operator registry or plugin ABI without concrete need.
Resolve static compatibility during preparation where practical, not repeated hot-path
lookup. Coarse dispatch at model/batch boundaries is acceptable.

Prioritize algorithm and data movement before micro-optimization: scalable prefill GEMM,
tiled attention, packed/ragged cross-request work, dynamic hybrid state, persistent request
rows and reusable metadata/workspaces. Async APIs alone do not overlap CPU/GPU work.
Captured graphs/fusion and alternative kernels are explicit qualified variants.

Current separate prefill/decode queues, one in-flight batch and full-reachable reservations
are prototype policy, not permanent requirements. Compare unified token scheduling only
when actual resource costs and mixed-arrival evidence exist. Speculation requires real
accepted/rollback ownership, not an enum label.

Measure load time, host/device peak memory, TTFT, inter-token latency, throughput, tails,
cancellation and stalled-consumer bounds. Distinguish trivial-executor host cost from
model/serving performance. Arithmetic-preserving changes require exact checks; reordered
algorithms need independently justified metrics and tolerances. Baseline error is not an
optimization budget. [Model qualification](../.agents/skills/model-integration/SKILL.md)
owns the recurring procedure.

## Composition, hardware and later training

Sequential encoder→decoder models can justify separate runtimes and a device-resident
handoff. Prompt-positioned VLM features may need coupled encoder/AR scheduling instead.
[Composition experiments](pipeline-composition.md) explain why neither a universal stage
graph nor one opaque multimodal request handle is established by current fixtures.

NVIDIA, AMD, Apple/Metal and CPU implementations may use different physical contracts.
Test a materially different backend before calling lifecycle abstractions general. External
systems allocate fleet resources; Ribn owns execution within its grant. Collectives,
sharding, expert/pipeline/context parallelism and disaggregated transfer require real
implementations and qualification, not topology metadata.

Coherent executable ownership comes before hot updates. An accepted request pins one
snapshot of weights/adapters/processors/numerical policy. Start drain-and-replace;
overlapping snapshots need explicit extra memory. Derived caches require semantic and
physical compatibility; exact snapshot identity is the safe initial policy.

Training is a later execution system, not an inference mode. Share artifact I/O, logical
identity, useful storage/completion and collective mechanisms where implementations align.
Gradients, autograd, activation retention, optimizer scheduling and checkpoint recovery stay
outside inference. Trainer-to-rollout publication and Python tensor/result interchange are
initial integration targets. Revisit shared model mathematics only when a real
forward/backward/update implementation demonstrates useful reuse or damaging duplication.

## Qualification breadth

Expand from current Qwen hybrid and BERT fixtures to a dense second decoder, real VLM,
encoder-decoder speech, iterative media model, non-AR text where useful and a second device
stack. Add structured decoding, logprobs, adapters, realtime/full-duplex sessions and wider
systems through concrete operation contracts. These are product targets, not a closed
universal list of runtimes or prerequisites for minimal serving.

The roadmap's next gates—not architecture breadth for its own sake—decide current work.
