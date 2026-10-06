# Current implementation

This is a source map, not a second target design. [Inference engine design](inference-engine-design.md)
owns direction; [resource protocol](resource-protocol.md) owns execution contracts;
[roadmap](roadmap.md) owns sequencing and open gates.

## Text generation

```text
CLI run
  → ribn-text: bounded preprocessing, chat/tokenization, incremental decoding
  → ribn::driver: one worker, request permits, owned streams, shutdown
  → ribn::Engine: AR lifecycle, scheduling, bounded output mailboxes
  → engine-qwen: QwenCuda / QwenExecution
  → engine-core: transitional batch/plan/logical-state contracts
  → engine-nvidia: physical state, dispatch and kernels
```

`TextOwner::load` assembles the Qwen GGUF/CUDA path. `TextOwner` owns shutdown;
cloneable `TextModel` handles share its token worker. Preprocessing uses a bounded
pool, not a second execution loop. Offline batching consumes a bounded window and
yields ordered per-item outcomes. A slow head bounds lookahead; decoding failure
abandons only its request. See [text semantics](resource-protocol.md#text-application-facade).

The token driver provides fail-fast application preparation permits, encoded-input bounds,
nonblocking bounded delivery, cancellation intent and retryable shutdown. Permits
cover preprocessing, execution and retained delivery, including retirement after
stream abandonment. Admission and AR preparation waits arm one-shot notifications on the capacity-one
wake channel before parking; only device completion retains timed polling.

Direct `Engine` use remains channel-free. It owns one in-flight batch and negotiates
aggregate work before launch. It validates the whole preparation report, refunds omitted
or shortened output credits and parks deferred rows outside runnable queues. All-omitted
offers rebuild for healthy unoffered peers before sleeping. Completion rows still validate
before any logical prefix or output commitment.
Cancellation is intent, not completion. Executor uncertainty faults the owner and
retains retirement ownership. Mailboxes can outlive execution slots; runtime-owned
discard works across that boundary, without frontend orphan queues.

Qwen reserves continuation for prompt plus output reach at admission. Deferred
admission retains no allocation and retries only after authority publication. It
checks whole-bundle capacity before allocation so a rolled-back partial attempt does
not wake itself. The adapter still translates scalar AR rows into transitional core
batch/state types; it has no second scheduler. Preparation retains one validated backend batch without taking
continuation leases; submit consumes its exact rows, and abandonment preserves old state.

Block-table attention reads exist in the backend, but production Qwen remains
contiguous. There is no paged write/growth consumer, prefix cache, restore operation
or recomputation preemption. The surviving `engine-core` contracts are transitional;
its duplicate scheduler, serving runtime and `ribn local` frontend have been removed.
The Qwen-owned serving benchmark now exercises the real AR engine and preserves
explicit GEMV variants; its new timings are not the legacy baseline's qualification.

## Encoder and artifact paths

`ribn-safetensors` validates artifact bytes once and exposes indexed tensor views.
`ribn-hf` resolves local config and unsharded/sharded weights without choosing model
semantics. It lazily caches whole owned shards; cache-entry limits do not bound peak
bytes while loading or while clones retain storage.

`engine-bert` owns configuration, parameter mapping, masked bidirectional attention,
embedding/LayerNorm/GELU/pooler semantics and a CUDA executor through `ribn-batch`.
The executor chooses a feasible FIFO prefix and concrete byte envelope. Results own
device storage, a producer completion dependency and the lease covering them together.
Dequeue does not refund the charge. Cancellation and malformed handoff return storage
through the executor's retirement path; uncertain enqueue quarantines storage and its
charge until a drain proves completion.

`ribn-foundation` owns `BytePool`/`PoolLease`/`AllocationId`, shared readiness,
parameter version labels and scalar metadata. Unused topology/materialization/placement
metadata and the fixture-only operator registry are removed. AR logical state and
encoder reservations use this same byte authority; host tests cover supplied pools,
atomic hybrid grants and sibling readiness. Production Qwen retains a private pool.
Shared CUDA physical retirement and downstream-workspace progress remain unproven.

## Source owners

| Path/package | Responsibility |
| --- | --- |
| `crates/runtime` / `ribn` | AR lifecycle, scheduling, mailboxes and owned token driver |
| `crates/text` / `ribn-text` | Text processor, bounded preparation/decoding, owned facade and offline batching |
| `crates/qwen` / `engine-qwen` | Qwen config, GGUF mapping, CUDA assembly and AR adapter; serving benchmark |
| `crates/nvidia` / `engine-nvidia` | NVIDIA storage, completion, dispatch and kernels |
| `crates/bert` / `engine-bert` | BERT semantics, preparation, executor binding and device retirement |
| `crates/core` / `engine-core` | Transitional execution/batch/plan, backend, state and weight contracts; no scheduler |
| `crates/gguf` / `engine-gguf` | GGUF artifact access and current tokenizer/chat-template implementation |
| `crates/safetensors` / `ribn-safetensors` | Generic validated SafeTensors views |
| `crates/hf` / `ribn-hf` | Local HF-style config/shard resolution and residency |
| `crates/foundation` / `ribn-foundation` | Byte accounting, readiness and consumed version/scalar types |
| `crates/batch` / `ribn-batch` | Non-AR batching, leased results and cancellation |
| `crates/cli` / `ribn-cli` | `inspect` and experimental `run` |

`tools/check-boundaries.py` checks production dependency direction and retired-path
resurrection. It does not freeze transitional crates against an accepted migration.

## Qualification and missing support

Current Qwen/text and BERT fixture evidence lives in [runtime qualification](../benchmarks/runtime-contract.md),
[encoder qualification](../benchmarks/encoder-qualification.md) and
[prefill qualification](../benchmarks/qwen-prefill-qualification.md). Host fixtures
establish lifecycle contracts, not GPU memory safety or model support. Migrated device
fixtures, notification changes and AR preparation still need re-execution when desktop is available.
GDN chunk scan remains opt-in while its full-model numerical gate fails.

[Foundation evidence](execution-foundation.md) and [composition counterexamples](pipeline-composition.md)
retain useful limitations. They do not qualify a universal execution API.

Not implemented: HTTP serving, general architecture/source resolution, remote HF
loading, real media processors, dynamic hybrid continuation, executable snapshot
replacement, a second hardware backend or distributed execution.
