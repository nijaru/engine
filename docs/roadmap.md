# Engineering roadmap

Status: ordered implementation and decision gates, priority updated 2026-09-22.

[Target design](inference-engine-design.md) owns architecture and API semantics;
[resource protocol](resource-protocol.md) owns execution ownership. This document
owns work order and unresolved decisions. Server-first inference is the priority;
training is a future execution system sharing demonstrated lower-level mechanisms.
All v0 interfaces may change. Do not preserve a broken API for compatibility.
These slices stage the [competitive product target](inference-engine-design.md#scope-and-success-criterion);
the initial Qwen/CUDA scope is not a narrower product strategy.

## Current execution order

The bounded CUDA Rust gate-2 experiment has a [defer verdict](cuda-rust-migration.md#gate-2-candidate-deferred-2026-09-22),
not acceptance: independent arithmetic and sanitizer checks pass on the tested
fixtures, but single-row projection and GDN regress against the qualified C++ path.
Preparation measurements do not offset those kernel regressions. Do not start gate 3
or expand this candidate into a compiler project. The migration owner records the
remaining coverage gaps and concrete re-entry conditions.

**Next: slice 4b/4c's coherent constrained-memory continuation path on the qualified
backend**, then slice 4d and minimal serving. Proceed in this order:

1. If representative kernels qualify, integrate a bounded slice through the existing
   execution owner under migration gate 3. Do not add a second serving loop or change
   resource ownership implicitly. A blocked migration does not block the existing
   qualified backend or justify waiting for full kernel coverage.
2. Finish slice 4b/4c's coherent constrained-memory continuation path: backend-owned
   blocks, growth preparation and safe retirement, a progress policy preventing
   all-active-growth deadlock, then valid hybrid prefix reuse and eviction/preemption.
   Slice 4d measures scheduling choices on real mixed workloads.
3. Bring forward slice 5's minimal serving protocol over the owned application API and
   matched single-device serving qualification. It does not wait for all of slice 4e's
   model/composition breadth or for complete CUDA Rust migration. Streaming, disconnect,
   overload, cancellation, bounded memory and lifecycle checks remain required.
4. Expand model coverage, migration and wider systems from measured results. Broad
   product ambitions remain unchanged; architecture breadth is not the next acceptance
   result.

Keep kernel-language qualification separate from inference competitiveness. Both need
integrated evidence; a faster isolated kernel or a successful Rust port does not prove
better model serving. The numbered slices below retain their dependency contracts and
historical evidence; this section owns which eligible work runs next.

## Engineering method

For each slice:

1. Inspect current source and relevant experiments. Research current primary sources
   for consequential uncertain choices; record revision/date and limitations.
2. State the contract in its existing owner: inputs/results, owner at each transition,
   cancellation and error scope, memory bounds, readiness and shutdown. Include the
   direct single-model path and a materially different counterexample.
3. Resolve competing designs with a small experiment when reasoning is insufficient.
   Do not mistake a fixture for model support or a prototype type for a stable API.
4. Write contract tests and implement a complete vertical slice. Replace obsolete
   paths instead of adding wrappers. Keep qualified kernels while replacing control
   plumbing unless kernel changes are independently required.
5. Run host checks and affected device gates, inspect actual user behavior and compare
   performance where the change affects it. Update current-state evidence only then.

Agents may choose local representation and algorithms within the contract. A change
in ownership, failure isolation, observable API semantics, bounds, execution order or
numerical policy requires a design/test update before dependent implementation.
Research does not need to settle every future feature before the first slice starts.
It must settle the slice's contract and show that known future requirements do not
contradict it. Record assumptions and reconsideration triggers, not absolute promises.

## Starting baseline

The historical starting point below explains the repair order. Slice entries record
later changes; [architecture](architecture.md) describes implemented surfaces.

At `2abf382`, Qwen GGUF/CUDA is the only qualified model execution path. The text
facade remains mutable and single-caller. The batch/BERT, topology and composition
paths are design experiments, not production model coverage.

- Progress completion allows partial prefill, but also incorrectly allows zero
  successful prefill. `Blocked` requeues immediately, potentially in the same step;
  there is no parked state or readiness protocol. Remove that incomplete API until
  real preparation supplies ordinary backpressure and reactivation.
- Text cleanup improved but is not complete: unknown execution identity can still
  have an output mailbox; batch/stream error paths can abandon cleanup. Move discard
  responsibility into the runtime rather than extending frontend retry lists.
- Batch terminal-count bounds were fixed with a failing-before regression. Output
  dequeue still does not establish a bound on downstream live physical allocations.
- Qwen reserves full continuation capacity and translates through legacy core types.
  Foundation version/topology metadata does not own executable storage.
- Device gates at `6f0ebbf`: lifecycle 4/4 (76.80 s), exact-token runtime (174.39 s),
  CUDA reference 61/61 (408.84 s). These qualify that revision's tested device paths,
  not missing lifecycle/readiness contracts. See [runtime evidence](../benchmarks/runtime-contract.md).

Detailed historical experiments remain in [execution foundation](execution-foundation.md)
and [pipeline composition](pipeline-composition.md). Performance/numerical evidence
belongs in [benchmarks](../benchmarks/README.md), not repeated milestone tables.

## Ordered vertical slices

### 1. Repair the execution ownership baseline

Partially implemented at `4477a0e`: runtime-owned discard replaces frontend cleanup
lists; batch and stream errors relinquish output interest. Host checks and CUDA text
lifecycle 4/4 passed. A failing-before test also pins cancellation through the legacy
blocked completion branch. See [qualification](../benchmarks/runtime-contract.md#runtime-owned-discard-qualification-2026-09-14).
At `14d0290`, the completion-time blocking API is removed and prefill advancement
must be positive. Completion validation no longer allocates/clones row plans and
retains batch-slot capacity. Zero-progress rejection has a failing-before regression;
partial-prefill runs with a two-credit output pool; mixed malformed rows, cancellation,
failed release and faulted retirement are covered. Host checks pass. Combined device
gates at `14d0290` pass: text lifecycle 4/4 (76.91 s), exact-token runtime (174.22 s),
CUDA reference 61/61 (408.13 s), each with recorded exit 0. The next design gate is
owned concurrent access. Direct host injection into the CUDA-gated text decoder and
broader frontend error-path testing remain part of that gate; current device tests
are narrow, not exhaustive frontend failure coverage.

Decided scope:

- retain positive contiguous partial-prefill completion and whole-batch validation;
- remove incomplete completion-time blocking and redundant completion-plan copies;
- runtime-owned cancel-and-discard, including terminal mailboxes after slot release;
- remove frontend cleanup queues and ensure every batch/stream error relinquishes
  request interest without releasing device-visible storage;
- distinguish request-local rejection from executor corruption. Conservative global
  faulting remains correct for unknown device state, not ordinary resource shortage.

Exit: failing-before regressions for zero progress, discarded terminal mailboxes,
late completion, mixed healthy/abandoned consumers and error cleanup; output-credit
accounting stays bounded; affected device gates pass. Do not build a synthetic
readiness framework merely to retain the old `Blocked` variant.

### 2. Owned concurrent application access

Decided contract: cloneable handle; one execution owner; owned request streams;
bounded admission and output; cancellation under saturated admission; explicit owner
shutdown/failure; one coherent executable lifetime. Direct runtime use remains valid.

First increment at `7005fad`: `ribn::driver` owns the existing token engine on one
worker, with fail-fast preparation permits, encoded-input envelopes, bounded owned
streams, cancellation/credit wakeups and explicit shutdown retry. Flume supplies
blocking and async waits without Tokio. A private runtime lifetime guard holds each
admission charge through retirement, including after stream drop. Exact ownership,
error and cancellation semantics are in the [resource contract](resource-protocol.md#owned-ar-driver-contract).

Nineteen actual-driver host tests pass, including 100 consecutive suite runs and
mutation checks for retirement charges, wakeups and queued acknowledgement teardown.
Required host checks pass. [Direct/driver host cost](../benchmarks/runtime-alignment/README.md#owned-token-driver-7005fad)
is measured; it is not model throughput. The CUDA device gate **passed on 2026-09-14**:
2/2 in 199.11 s on the idle RTX 4090 with the pinned artifact, including
`owned_driver_preserves_reference_with_stalled_and_abandoned_peers`. See the
[device evidence](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).
The matched direct-versus-handle comparison is host-only at the driver level; the text
facade adds a device comparison below.

#### Audit alignment order (2026-09-14)

The [source-traced assessment](architecture.md#slice-2-boundary-audit-2026-09-14-84a146a)
found text-boundary gaps, not a reason to replace the token driver. Its findings are
resolved by the owned text facade; the [text contract](resource-protocol.md#text-application-facade)
is now implemented and host-qualified.

1. **Done (2026-09-14).** Desktop was reachable and idle; the checkout was pulled and
   the pinned artifact hash re-verified. Serialized gates, all exit 0: runtime 2/2
   (199.11 s, including the stalled/abandoned-peer driver test), text lifecycle 5/5
   (95.83 s), and a new multi-request determinism test 1/1 (19.52 s). The concurrency
   example ran four callers end to end. Evidence:
   [runtime contract](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).
2. **Done.** The text contract landed before dependent code: a bounded preprocessing
   pool with its own shutdown owner, per-item settlement instead of whole-input
   preparation atomicity, UTF-8/terminal precedence, typed error sources, and
   growth-time byte limits for input/rendered/prompt/decoded payloads. Overload stays
   fail-fast per item with no waiter queue and no spin-retry.
3. **Done.** The borrowed execution loop is replaced by the
   host-testable owned facade with CUDA-only assembly. Host tests cover decode failure
   with a healthy peer, cancellation, dropped streams and batches, a stalled consumer
   beside a healthy request, buffered owner-failure semantics at the driver level, and
   incomplete UTF-8 on a normal terminal. Driver-level tests retain shutdown retry
   ownership.
4. **Done.** Eager batching and its atomic-preparation test were replaced together. A
   pull-counting iterator proves bounded lookahead, and per-item failures — an invalid
   input, a decode error and saturated admission — leave peers unaffected.
5. **Partially done.** CLI cutover, explicit shutdown and host/CUDA-feature checks are
   complete, collect-result exclusions are documented, and affected device gates pass
   (item 1). CLI file/stdin ingestion is now bounded before loading by the processor's
   input allowance, including chat-role bytes, with a one-byte overflow probe. Host
   reader tests and CUDA-feature CLI process tests cover exact/oversized input.

Text facade status: `ribn-text` now provides cloneable `TextModel` handles over one
owned driver, `TextOwner` shutdown, bounded preprocessing, typed `TextError` sources,
owned `TextStream`, and ordered bounded-window `TextBatch`. Twenty-two host tests pass
with a real GGUF tokenizer, the real driver and a scripted fixture device, including
per-item overload under full permit retention, bound-checking at the retention
boundary, owner-failure delivery and a dead preprocessing pool that fails callers
instead of queueing them. Six CUDA-backed tests pass on the device, including
multi-request determinism. Evidence:
[runtime contract](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).

The chat stop-token set is decided and implemented; see
[the policy and its evidence](../benchmarks/runtime-contract.md#chat-stop-policy-2026-09-14).
It also governs the legacy `ribn local` path. Reasoning controls remain unexposed.

Matched frontend overhead is measured on the device: at concurrency 1 the handle path
adds a fixed 5–12 ms per request plus about 0.3–0.5 ms per delivered token, under 0.5%
of end-to-end time once the prompt carries real work. See
[CUDA frontend overhead](../benchmarks/runtime-alignment/README.md#owned-text-facade-over-cuda-d61b62c).

Remaining, and not required to close this slice:

- thread-affine non-Send construction, if a real backend requires it, needs a separate
  factory contract;
- reasoning controls remain unexposed, so thinking stays fixed off in the text facade;
- matched concurrency-above-1 and mixed-arrival frontend comparison belongs to the
  serving slice, not to this single-request measurement.

Exit: device-qualified frontend behavior, saturation/race/shutdown evidence on both
paths, no orphan requests, and matched frontend overhead measurements. The loaded owner
pins one actual executable configuration; no metadata-only snapshot wrapper or
unimplemented general architecture registry counts as snapshot ownership.

Slice status: **closed at 2026-09-15** (`d61b62c`), except for the explicitly
conditional items above. The remaining work moves to slice 3.

### 3. Real asynchronous encoder and prepared resources

Decided contract: [encoder preparation and prepared
resources](resource-protocol.md#encoder-preparation-and-prepared-resources) — one byte
pool authority granting owning leases, accepted ranges rather than per-row budgets,
waiting distinguished from permanent rejection with a real readiness source, pre-submit
abandonment releasing all new reservations, partial-enqueue ownership, and a
device-resident result carrying its lease and producer completion dependency.

The first concrete representation is deliberately not generic. Increments:

- **3a (done 2026-09-15).** `ribn-foundation` owns the first concrete authority
  (`BytePool`/`PoolLease`/`AllocationId` in `crates/foundation/src/pool.rs`): owning
  non-duplicable leases, an allocation identity per grant, and a release epoch as the
  capacity readiness source. `ribn-batch` reserves retained output through the pool
  instead of a private counter, so one constraint binds every runtime drawing on it;
  `Rejection::RetainedOutputTooLarge` replaces the per-runtime byte bound, and a failed
  submission returns its leases inside the error rather than silently releasing bytes
  the device may still cover. Host evidence: 13 `ribn-batch` unit tests and 11
  `ribn-foundation` tests, including over-grant refusal, exact-fit grants, charge
  survival across handoff and dequeue, sibling binding, epoch advance on release,
  oversized-head rejection with a healthy peer, closed-pool refusal, and lease retention
  across a failed submission. Workspace checks pass (`boundaries`, `fmt`, 49 test suites,
  `clippy` default and CUDA).
- **3b.1 (done 2026-09-15).** A real device encoder now exists: `engine-nvidia` owns the
  encoder primitives (embedding summation, mean-centred LayerNorm, exact-erf GELU,
  masked attention with a stable softmax, row bias, residual add, tanh) and
  `engine-bert` owns the model path — configuration, parameter mapping over a resolved
  package, cuBLAS projections, and an owned submission that holds its device buffers
  and a recorded completion event. Qualified against an independent Hugging Face
  reference: worst absolute deviation 4.77e-7 on hidden states and 1.77e-8 on pooled
  output across four cases including masking and segment types. See
  [encoder qualification](../benchmarks/encoder-qualification.md). The path is fp32
  and fixture-geometry only; a production-size encoder must be re-qualified at its own
  geometry.
- **3b.2 (done 2026-09-15).** The device encoder runs through the batching runtime
  rather than a second scheduler. `engine-bert` decides its accepted range and its
  byte envelope on the host (`crates/bert/src/request.rs`: concrete shapes, structured
  request-local constraints), implements `ribn-batch`'s executor seam
  (`crates/bert/src/executor.rs`: `select_batch` accepts the executable prefix,
  `retained_bytes` reports `request_bytes`, `execute` enqueues without waiting), and
  hands a completion to its consumer as an `EncoderResult` that owns the device storage
  and the pool charge covering it together, behind an explicit producer completion
  dependency. `EncoderSubmission` no longer borrows its encoder — its prepared
  resources are shared — so a result outlives the submitting call, and
  `device_bytes` counts the allocations that exist rather than restating the
  prediction. `ribn-batch` publishes `CapacityWait`, a registration over the pool's
  release epoch, so a parking caller registers before attempting and rechecks after a
  bounded park while the pool still publishes no wakeup. Evidence: the encoder's
  parity, envelope and pool-bound device run in
  [encoder qualification](../benchmarks/encoder-qualification.md), plus host lifecycle
  tests over the real `BatchRuntime`/`BytePool` for constrained capacity, deferred
  completion that must be awaited before reading, a charge surviving dequeue, a stalled
  consumer beside a healthy peer, and a rejected oversized request with a progressing
  peer. A failed submit drains the stream before its buffers drop; that best-effort step
  is replaced by 3b.3's handshake.
- **3b.3 (done 2026-09-15).** Partial-enqueue ownership is the executor's, and the
  error-carried lease placeholder is gone. `ribn-batch`'s `Job` carries the reservation
  granted for its request, so the charge travels with the storage the executor
  materializes; `BatchExecutor::execute` reports `EnqueueError::Refused` (nothing
  reached the device) or `Uncertain` (a device-visible mutation happened), and the new
  required `retire` hands output the runtime cannot commit back to the executor.
  `RuntimeError::Executor`/`Uncertain` carry requests and a source, never reservations,
  and `MalformedCompletion` reports what came back instead of holding a charge.
  `engine-bert`'s encoder owns a retirement list holding each quarantined submission
  with the reservation covering it: `submit` keeps a fully constructed submission there
  after a failed enqueue, `drain_retirement` is the only release, and when the drain
  fails the encoder records a fault and refuses new submissions until one succeeds.
  Dropping a device buffer was never the hazard — cudarc frees in stream order or drains
  first — so the reason to hold the charge is accounting: a stream-ordered free has not
  returned the memory to the allocator, and releasing early would let another owner
  reserve bytes the device is still holding. Evidence: host tests over the real runtime
  and pool for clean refusal, uncertain failure (charge kept with the executor, released
  only by a proven drain), uncommittable output handed back, and a mocked device
  completion; device evidence for the encoder's own retirement cycle in
  [encoder qualification](../benchmarks/encoder-qualification.md). Failure injection
  exists because only an OOM or a device fault reaches that path in production.
- **3c (done 2026-09-15, `3fc3b2f`).** The non-AR runtime had no cancellation path, so this
  increment added the one its exit criteria require. `BatchRuntime::cancel` records intent
  against a live request: a waiting request keeps its FIFO place and is reported instead
  of executed, so no second cleanup list is needed and the intent stays deliverable
  while the terminal count bound is full; a request whose result is already retained
  hands that result, and the charge covering it, back through `BatchExecutor::retire`.
  `Terminal::Cancelled`/`StepOutcome::Cancelled` are the notifications, the accepted
  range stops before a cancelled request, and the runtime never releases device storage
  or its charge on its own. Device qualification at `3fc3b2f` (7 encoder tests plus 3
  parity tests, serialized on an idle RTX 4090) covers the constrained shared pool with
  the real device envelope, a lagging consumer that holds its charge while a peer keeps
  progressing, cancellation before enqueue (waiting for capacity) and after enqueue
  (a retained result handed to the executor, which is what a handoff the runtime cannot
  make looks like), a handoff the consumer simply never takes (charges surviving dequeue
  until that consumer releases them and then released together), permanent rejection both
  by model shape and by pool capacity with a peer progressing, and the encoder's own
  retirement cycle behind an injected post-enqueue failure. See
  [encoder qualification](../benchmarks/encoder-qualification.md).

  Two caveats are recorded rather than claimed. A *slow device* cannot be scheduled
  deterministically at this geometry, so "delayed completion" is qualified as the
  property that matters — no runtime or submission path waits for completion before
  handing a result over, a result whose completion is not established can only be read
  after the consumer awaits the dependency, and a lagging consumer never blocks a
  peer — instead of by timing a stalled kernel. "Downstream workspace stays available"
  is not yet provable: no downstream stage shares this pool until slice 4 wires the AR
  path to it, and the sibling-progress case is what stands in for it today.

Exit: constrained shared pool, delayed completion, cancellation, failed handoff,
consumer stall and permanent oversized-input rejection with healthy peer progress.
Charges survive dequeue until safe reuse. Prove downstream workspace remains available.
Do not implement a universal cost vector or operation graph as a prerequisite.

### 4. Dynamic hybrid AR and model-local integration

Implement dynamic KV plus recurrent-state ownership, valid prefix reuse, eviction and
preemption before speculative reconciliation. Compare unified token-budget scheduling
with current queues using actual resource costs and mixed-arrival workloads. The
contract they must satisfy is [AR continuation resources](resource-protocol.md#ar-continuation-resources);
this list is the order it is implemented in. Each step lands green and device-checked,
because a continuation change silently corrupts output when it is wrong.

- **4a. Continuation capacity is per request (done 2026-09-15, `323f258`).** A
  declaration is a shape plus a bound; a request materializes the capacity it can reach
  (prompt plus output budget) and is charged for that, not for the model's context. The
  prototype reserved full maximum context for every admitted sequence, so a short request
  spent the same device bytes as a long one and concurrency was bounded by context, not
  by demand. Insufficient capacity is now waitable backpressure (`Admission::Deferred`)
  rather than a request failure, while a demand the authority can never grant stays
  request-local rejection. Evidence: host tests for bound acceptance, over-bound and
  shape rejection, capacity waiting and infeasible demand; device qualification on the
  pinned Qwen artifact at `323f258` — three concurrent requests run inside a 500 MiB
  authority that holds fewer than two context-sized charges (849 MiB) and each still
  produces the exact reference output, alongside the unchanged AR driver and text
  lifecycle gates.

  The measurement also sizes 4b for this model: the pinned hybrid's 16 full-attention
  layers cost 16.8 MiB of KV at a 257-token reach and 275 MiB at 4096 tokens, while its
  recurrent state costs a fixed 149.6 MiB per sequence. A per-request charge is therefore
  166.8 MiB against a 424.6 MiB context-sized one, and the recurrent state — not the KV —
  dominates continuation memory. Reusable hybrid boundaries cannot checkpoint that state
  at every block; 4b must decide the coarsest boundary that still pays.
- **4b. Block-granular continuation with a reusable prefix cache.** Continuation is
  allocated in fixed token blocks with content-derived identity, growth reserves blocks
  as the accepted range advances and settles them at the committed boundary, and a block
  shared by several sequences is owned once by the cache. For a hybrid model a reusable
  boundary additionally requires its recurrent checkpoint, so a KV-only match is not a
  match. Evidence: parity with and without a reused prefix, identical output for a shared
  prefix across sequences, and a constrained-pool run where reuse admits work that
  per-sequence allocation cannot.

  First prerequisite: admission now retains an authority-owned readiness registration
  and retries only after publication, not on every decode step. Qwen registers before
  checking whole-bundle capacity; failed hybrid allocation must not roll back a partial
  reservation and wake itself. Host regressions cover unchanged capacity, successful
  versus failed release, publication before driver parking without an in-flight batch,
  and cancellation of a registered wait. The driver still observes epochs with its
  bounded timed fallback; there is no resource-to-worker notification yet. Block
  allocation, growth preparation and prefix reuse remain unimplemented. Device
  requalification passed at `c9253bc`: the three CUDA runtime gates passed serially
  on the RTX 4090, including constrained admission and stalled/cancelled peers.

  Implementation order within 4b:
  1. **Done (2026-10-12, `ee689d7`).** Block-table KV addressing is qualified against
     contiguous attention with unchanged arithmetic: scores and outputs are bit-identical
     for shuffled and identity tables across aligned blocks, partial tails and causal
     multi-row prefill, and unsatisfiable geometry is rejected before launch. The
     contiguous entry point and the 61-test kernel suite are unchanged, and the pinned
     Qwen device gate still passes. Evidence:
     [block-table KV addressing](../benchmarks/runtime-contract.md#block-table-kv-addressing-2026-10-12-ee689d7).
     This backend primitive alone enables no runtime paging or prefix reuse; production
     Qwen still holds one contiguous allocation per sequence.
  2. Replace contiguous sequence storage with backend-owned blocks and prepare aggregate
     growth before launch. New reservations settle only at validated completion;
     failed partial enqueue retains the old continuation and new growth together.
  3. Integrate growth waiting outside runnable queues, refunding unused output credits.
     Before admitting partial envelopes, resolve the all-active-growth deadlock:
     several sequences may retain the entire pool while none can finish. Registered
     readiness is insufficient; protected completion headroom or recomputation
     preemption must accompany that change (even though preemption was listed in 4c).
  4. Add snapshot-scoped content-keyed immutable blocks and sparse recurrent checkpoints.
     Restore only a complete hybrid boundary, reserve a private writable recurrent
     copy, and replay when no matching checkpoint exists. Checkpoint spacing and block
     size remain measurement decisions, not global model semantics.
  5. Run shared-prefix parity and constrained-pool admission through the real Qwen
     executor, plus cancellation, abandoned preparation and failed-retirement gates.
- **4c. Eviction and preemption.** Unreferenced cached blocks are evicted
  least-recently-used; a live sequence whose growth cannot be granted is preempted by
  recomputation before any host swap tier exists. Evidence: constrained-pool
  qualification of eviction, recompute identification of that sequence's own charge,
  cancellation under saturation, and no leaked charge after either path.
- **4d. Scheduling comparison with real costs.** With per-step preparation reporting
  actual demand, compare a unified token budget against the current prefill/decode
  queues on mixed-arrival, mixed-length workloads and report which is which: prefill and
  decode remain execution distinctions either way. Evidence: matched repeated
  measurements with the same artifact and numerical policy, plus the policy choice and
  its losing case recorded here.
- **4e. Composition.** A second real decoder and a real VLM/processor integrate through
  model/processor, backend, registration and tests—not family branches in cancellation,
  routing or protocols. A sequential encoder-decoder and a small iterative model then
  test composition beyond AR without forcing every encoder into its own stage. Evidence:
  independent numerical references and, for the encoders, the existing slice-3
  qualification path.

Exit: independent numerical references, constrained resources, continuation and
cancellation qualification; documented shared changes and their concrete necessity.
Replace legacy Qwen/core translation as its real consumers migrate. Delete unused
paths immediately; do not wait for every future runtime class to remove dead code.

### 5. Serving qualification and wider systems

Bring the minimal single-device serving surface and comparison forward after the
constrained-memory continuation/scheduling gates; slice 4e's broader composition and
complete CUDA Rust migration are not prerequisites. Wider-backend/distributed claims
still require their own gates below.

Implement a documented protocol subset over the owned application API, not another
execution loop. Test streaming, errors, disconnects, overload, health/readiness,
metrics and security limits. Python in-process use follows the same semantics.

Qualify mixed lengths/arrivals, long context, stalled clients and memory pressure.
Report throughput within explicit latency objectives, latency distributions, host
cost, peak memory and cancellation latency. A benchmark needs pinned workload,
artifact, numerical policy, revisions and repeated matched measurements. Compare
against vLLM/SGLang or another relevant serving baseline on overlapping supported
workloads with matched hardware and quality settings; report wins, regressions and
unsupported scope separately. Internal speedups alone do not establish competitiveness.

Test a materially different backend before calling device contracts general.
Implement collectives and real sharding before promoting topology metadata into a
distributed runtime. Local single-device execution must not require serialization
or synthetic worker/process layers.

### 6. Training integration, then training execution

First establish trainer-to-rollout version publication and tensor/result interchange.
Start drain-and-replace; overlapping snapshots require explicit extra memory and
compatibility evidence. Never expose optimizer-mutating storage to inference.

A later small forward/backward/update reference test determines shared operator/model
semantics. Training owns gradients, activation lifetimes, optimizers and its scheduler.
Checkpoint/restart and distributed training require their own qualification. Revisit
shared model representation if duplicated semantics becomes a measured maintenance or
correctness problem; do not impose autograd on inference preemptively.

## Independent qualification tracks and blockers

- **GDN scan:** opt-in only. Full-model gate fails `1.10e-2` versus `5.0e-3`; cause
  unestablished. Diagnose captured real state against a higher-precision recurrence.
  Existing baseline reference error grants no tolerance budget. Compare common
  histories; step 19 after step-18 token divergence is not a common-history comparison.
- **Kernel scaling:** retain qualified small-M GEMV; investigate tiled quantized GEMM,
  packed cross-request work and tiled attention. Do not repeat measured losers without
  new evidence: four rows per warp, shared IQ4 codebook, pre-elimination decayed keys.
  Compilation is not GPU qualification; CUDA Rust migration has its own
  [gate](cuda-rust-migration.md), not priority over runtime correctness.
- **Artifact loading:** GGUF lazy readers reduced sampled descriptors 888→38. HF
  shard-cache count is not a hard peak-byte limit; incoming loads and retained clones
  can overlap. Qualify streaming/byte-aware ownership before large-model claims.
- **Research:** [research agenda](research-agenda.md) supplies candidate sources and
  later topics. Only the relevant questions above block a slice; do not turn broad
  research into a second implementation backlog.

## Required checks

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

Capture actual exit codes, including when logging through pipes. CUDA-feature checks
compile otherwise gated frontend code but do not replace serialized device tests.
Follow the [model-integration skill](../.agents/skills/model-integration/SKILL.md)
for model/numerical qualification. Record failed and unrun gates explicitly.
