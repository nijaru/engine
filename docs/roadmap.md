# Engineering roadmap

[Target design](inference-engine-design.md) owns product direction and architecture;
[resource protocol](resource-protocol.md) owns execution/resource ownership;
[architecture](architecture.md) maps current source. This document owns work order,
unresolved decisions and acceptance gates. All v0 interfaces may change.

## Current execution order

**Next: the coherent constrained-memory AR continuation path (4b/4c), then scheduling
qualification (4d) and minimal serving (5).** Broader model/composition coverage (4e)
and complete CUDA Rust migration are not prerequisites for minimal serving.

The [CUDA Rust gate-2 candidate](cuda-rust-migration.md#gate-2-candidate-deferred-2026-09-22)
is deferred: tested arithmetic/sanitizer gates pass, but single-row projection and GDN
regress against the qualified C++ path. Preparation measurements do not offset those
regressions. Do not start gate 3 or turn the experiment into a compiler project. A
blocked kernel-language migration does not block runtime work on the existing backend.

Desktop unavailability permits host implementation and CUDA-feature compile checks,
not device qualification. Keep the qualified contiguous/full-reachable Qwen policy
until paged writes, growth, restoration and retirement pass their device gates. Host
fixtures may establish control contracts without enabling an unqualified physical path.

## Completed foundations

| Slice | Result and retained evidence |
| --- | --- |
| 1. Ownership baseline | Positive partial progress, whole-completion validation, runtime-owned discard and retryable retirement. Completion-time blocking was removed. [Runtime evidence](../benchmarks/runtime-contract.md) |
| 2. Concurrent application access | One token execution worker, cloneable bounded handles/streams, cancellation and shutdown retry; bounded text preprocessing/decoding and ordered offline batching. [Contract](resource-protocol.md#owned-ar-driver-contract), [frontend overhead](../benchmarks/runtime-alignment/README.md) |
| 3. Real encoder preparation | BERT CUDA fixture through `ribn-batch`, shared byte leases, completion-owned results, uncertain-enqueue retirement and cancellation. [Encoder qualification](../benchmarks/encoder-qualification.md) |
| 4a. Request-reachable continuation | Qwen charges prompt plus output reach, not maximum context. Three reference-matching requests fit a 500 MiB authority that cannot hold two context-sized charges. [Runtime evidence](../benchmarks/runtime-contract.md) |
| 4b host preparation | Aggregate accepted ranges, credit refunds, parked readiness, local rejection and uncertain-retirement ownership through the real AR engine. Qwen validates/retains a full-reachable prepared batch; no dynamic physical growth. [Host gate](../benchmarks/runtime-contract.md#ar-pre-submit-preparation-host-gate) |
| 4b host authority | AR reservations use supplied `BytePool` authorities with one atomic hybrid grant, sibling readiness and closure handling. Production Qwen keeps a private pool. [Host gate](../benchmarks/runtime-contract.md#ar-reservation-authority-host-gate) |
| 4b prerequisite | Block-table attention **reads** qualified at `ee689d7`; production Qwen remains contiguous. [Addressing evidence](../benchmarks/runtime-contract.md#block-table-kv-addressing-2026-10-12-ee689d7) |

Admission notification (`3e4dbec`) and shared batch/AR readiness (`e8e3b48`) are
host-qualified; device requalification is pending. The driver arms one-shot resource
notifications on its bounded wake channel; only pending device completion uses timed
polling. Idle admission no longer polls. Pool closure wakes waiting consumers without
refunding live leases.

Cleanup removes the duplicate core scheduler/runtime and `ribn local`, retaining the
Qwen/backend bridge and moving its benchmark onto the real AR engine. Legacy host
lifecycle tests retire with that implementation; current runtime/Qwen tests retain
ownership protection. Migrated 8/9-row physical-retirement fixtures and the new nine-row
AR cancellation case require device execution. Historical benchmark measurements remain
historical; the migrated harness has no inherited performance qualification.

## 4. Dynamic hybrid AR and model-local integration

### 4b. Block storage, preparation and prefix reuse

Source review found that the previous five-step plan omitted physical backing,
replay ownership and restore-to-logits semantics. Resolve those before dependent
paging code; a scalar prefix or block hash is not an allocation owner.

1. **Choose physical backing and its accounting.** The qualified block-table kernel
   indexes one shared contiguous K/V base per layer, not a vector of independent device
   allocations. Decide a bounded resident arena or qualify another addressing scheme.
   Arena bytes stay charged while resident; reusable slots have a separate occupancy
   authority and cannot refund physical bytes. AR now uses the encoder's `BytePool`
   mechanism with host-tested supplied authorities and atomic bundle grants. Production
   Qwen still owns a private pool; qualify physical retirement before sharing it with
   encoder execution or claiming a bound across runtimes on one device. Include paged
   **writes**, block-table transfer lifetime and all prepared lanes; read addressing alone is insufficient.
2. **Integrate physical growth with AR preparation.** The host control seam now accepts
   positive ranges, refunds unused credits, parks unaffordable rows with readiness and
   rejects impossible rows locally. Whole-report validation and abandoned/uncertain
   growth ownership pass through the real engine and byte authority. An all-omitted
   offer cannot strand an unoffered healthy peer. Qwen retains its validated backend
   batch but keeps full-reachable reservation; integrate real growth and retain that
   policy until the affected device gates pass.
3. **Prove progress before partial envelopes.** Active sequences can exhaust the pool
   while all wait to grow. Readiness cannot break this deadlock. Resolve protected
   completion headroom versus recomputation preemption. Recompute needs one bounded
   owner of prompt plus committed generated history, physical replay progress separate
   from delivered progress, no duplicate output/usage/resampling, and survivor growth
   priority over victim readmission. Reject requests whose own required state cannot
   fit rather than endlessly preempting them. Reclaim only completion-safe state.
4. **Add snapshot-scoped immutable content blocks and sparse recurrent checkpoints.**
   A reusable boundary requires all ancestor KV blocks and every hybrid component,
   not just a matching hash. Restore a private writable recurrent copy. Initially cap
   restoration strictly before the final prompt token and execute the suffix for logits;
   caching logits/output state is deferred. Replay when no complete boundary exists.
5. **Qualify through real Qwen execution.** Compare common histories with paging on/off
   and reuse on/off, including exact full-prompt hits, partial tails, missing ancestor
   blocks/checkpoints, cancellation during restoration, abandoned preparation and failed
   retirement. Demonstrate shared-prefix reuse admitting work that dynamic per-sequence
   allocation cannot, using a geometry where checkpoint economics permit it. Record
   checkpoint/copy peak bytes and replay/TTFT benefit separately.

Open layout/policy choices: arena size/growth, block size, checkpoint spacing,
completion headroom/preemption policy and concrete physical growth reservations.
The single-owner preparation transaction is defined in the resource protocol. Avoid a universal continuation framework or cost vector.

Recorded hybrid geometry: 149.6 MiB recurrent state per live sequence, 16.8 MiB KV at
257-token reach and 275 MiB KV at 4096. Three private recurrent copies plus even one
checkpoint exceed the recorded 500 MiB pool before KV. Dense checkpoints and a blanket
short-request concurrency improvement claim are therefore unjustified. Choose and
measure a longer shared-prefix workload; compare paging **with versus without reuse**,
not only against the older full-envelope implementation.

### 4c. Eviction and preemption

Evict unreferenced cached slots in LRU order. Physical arena charges release only when
arenas retire safely. Live-sequence recomputation begins with 4b's progress gate, not
a later optional cleanup. Keep cache eviction, request preemption and cancellation
distinct. Qualify constrained-pool progress, retained charges during device access,
replay correctness and complete recovery of capacity.

### 4d. Scheduling with real costs

Compare unified token-budget scheduling against current prefill/decode queues on mixed
lengths/arrivals, memory pressure and latency objectives. Prefill/decode remain execution
distinctions. Pin workload, numerical policy and revisions; repeat matched measurements
and retain the losing case as well as the selected policy. A single prompt sweep is not
representative serving evidence.

### 4e. Model-local integration and composition

A real second decoder and real VLM/processor must integrate through model/processor,
backend, registration and tests, not family branches in unrelated lifecycle/routing.
Then test a sequential encoder-decoder and small iterative model. Let actual models
determine staged versus coupled execution. Independent numerical references and the
encoder preparation path are required. Prove downstream workspace headroom under a
shared-pool constrained test; slice 3's sibling-progress test does not establish it.

## 5. Minimal serving, then wider systems

Implement a documented protocol subset over the owned application API, not another
execution loop. Test streaming, disconnects, overload, cancellation, health/readiness,
metrics, security limits and shutdown. Python in-process access follows the same semantics.

Qualify mixed arrivals/lengths, long context, stalled clients and memory pressure. Report
throughput within explicit latency objectives, latency distributions, memory peaks, host
cost and cancellation latency. Compare vLLM/SGLang or a suitable overlapping baseline
with matched hardware/quality, pinned artifacts/configurations and repeated runs. Report
unsupported scope separately; internal speedups do not establish competitiveness.

A materially different backend, real collectives/sharding and state transfer need their
own gates before portability/distributed claims. Single-device use must not require
serialization or synthetic process layers.

## 6. Training integration, then execution

Start with coherent trainer-to-rollout publication and Python tensor/result interchange.
Drain-and-replace precedes overlapping snapshots; overlap requires explicit capacity and
compatibility evidence. Never expose optimizer-mutating weights to inference.

A reference-backed forward/backward/update experiment determines useful shared model
mathematics. Training owns gradients, activations, autograd and optimizer scheduling.
Checkpoint recovery and distributed training require separate qualification.

## Independent blockers and conditional work

- **GDN scan:** remains opt-in. Full-model error `1.10e-2` exceeds `5.0e-3`; cause
  unestablished. Compare captured common-history state against higher-precision recurrence.
  Existing baseline error grants no tolerance budget.
- **Kernel scaling:** retain qualified small-M kernels. Investigate tiled quantized GEMM,
  packed work and tiled attention with actual measurements. Do not repeat measured losers
  without new evidence: four rows per warp, shared IQ4 codebook, pre-elimination decayed keys.
- **Encoder accounting:** partial allocation/upload construction and result/quarantine
  teardown need charge retention through proven backing release. cudarc stream-ordered
  free is not instantaneous physical capacity for sibling streams. Existing post-enqueue
  fault injection does not cover partial construction. Locked cudarc 0.19.9 can also
  lose the allocated pointer when tracking-event creation fails: a stream drain cannot
  release it. Resolve that failure owner and add partial-construction/teardown gates
  before a hard cross-stream peak-memory claim. See the
  [physical accounting contract](resource-protocol.md#encoder-preparation-and-prepared-resources).
- **Artifact loading:** HF shard-cache count does not bound peak bytes; incoming loads and
  retained clones overlap eviction. Establish streaming/byte-aware ownership before
  large-model claims.
- **Text access:** thinking controls, non-Send/thread-affine construction and higher-window
  offline throughput remain conditional work, not reasons to replace the existing owner.
- **Research/reference backend:** evaluate only when a concrete model/backend decision
  needs it. Architecture breadth is not a second implementation backlog.

## Required checks

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

Capture actual exit codes. CUDA compilation does not qualify device execution. Follow
[model integration](../.agents/skills/model-integration/SKILL.md) for numerical gates and
[execution contracts](../.agents/skills/execution-contracts/SKILL.md) for lifecycle changes.
Keep unavailable gates explicit rather than relabeling a host fixture as device evidence.
