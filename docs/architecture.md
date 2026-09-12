# Architecture

Ribn is a Rust-first inference runtime. The ground-up target is defined in
[ground-up design](ground-up-design.md); this document describes the implemented
boundaries. [Roadmap](roadmap.md) lists the remaining proof and retirement gates.

## Public interfaces and internal execution

The product target is familiar inference-engine CLI, HTTP, and idiomatic Rust
library interfaces, with sensible defaults and explicit tuning controls. Loading,
generation, batching, and streaming should be ordinary operations, not require
users to understand prepared tasks or executor selection. Task selection is
explicit only when the requested operation and model leave a real ambiguity.

The current `GenerationExecutor`, `BatchItem`, and mailbox APIs support runtime
and model integration. They are not a requirement for the eventual high-level
API to expose the same workflow. Internal backpressure should appear as normal
streaming and capacity/error behavior. Keep useful lower-level access for advanced
integrations, including device/resource control where supported; hiding that
control is not an architectural goal. The
[public-interface direction](ground-up-design.md#public-interface-direction)
defines the plan; it does not add capabilities to the current CLI or library.

## Generation runtime, not a universal model graph

`crates/runtime` exports `ribn`. The current task is token generation. Its
`GenerationExecutor` contract is explicit about that scope: admit a sequence,
submit prefill/decode work, observe accepted completion, release resources, and
synchronize shutdown. Other tasks may share infrastructure without using this
exact state machine. No arbitrary tensor graph, model-family enum, or device
allocation appears in the common request interface.

`Engine` owns persistent request slots and lifecycle. Internal configuration,
scheduling, completion validation, and output modules share that one owner.
`ExecutorInfo` describes the prepared executor's display identity and
`GenerationLimits`; it is not a persistent compatibility fingerprint.

## Dependency and ownership direction

| Package | Responsibility |
| --- | --- |
| `ribn` (`crates/runtime`) | Generation requests, admission, scheduling, committed results, output, cleanup ownership |
| `engine-qwen` | Artifact-independent `QwenConfig`, `QwenGguf` mapping, prepared `QwenCuda` executor |
| `engine-gguf` | Generic GGUF metadata, tensor reading, and tokenizer adapter; no Qwen model provider |
| `engine-nvidia` | CUDA mechanisms, kernels, current specialized Qwen execution, and NVIDIA submission adapter |
| `engine-core` | Legacy execution/state contracts and old runtime retained for comparison/cutover |
| `ribn-cli` (`crates/cli`) | `ribn inspect`, experimental `ribn run`, and legacy `ribn local` |

The model definition can build with no GGUF/CUDA features and no normal
third-party dependency. Serialized `qwen35.*` keys remain unchanged inside the
Qwen GGUF adapter. Backend specialization is legitimate; model geometry and
checkpoint container identity are not the same concept.

NVIDIA glue no longer lives in core. Development-only reference tests may depend
on higher-level adapters; production dependencies must retain the direction
checked by `tools/check-boundaries.py`. The legacy core is not the API to extend
when adding another generation model.

## Request, state, and completion invariants

`RequestId` is the user-visible identity; `SequenceId` identifies executor-owned
continuation state. Both are engine-issued and process-unique, not allocation
pointers or cache identities. Concrete model implementations retain strongly
typed state and model-owned resources separately. Neither recurrent state nor
host lookup resources must masquerade as KV pages.

The committed prefix counts consumed model inputs. Generated output is separate:
final prefill can emit output without consuming that token, and decode can return
bounded multi-token accepted progress. Draft/rejected work never appears as a
committed result. Fresh admission creates prefix-zero state only; restore, fork,
transfer, and reconstruction need actual compatible contents and completion proof.

Submission may enqueue device work. The entire completion batch is validated
against saved sequence identities, prefixes, and budgets before any logical state
or output is changed. Scheduling-policy updates do not invalidate earlier work.
In-flight cancellation suppresses uncommitted results after completion; it does
not release live buffers or cancel peers.

An execution/completion fault fences new submissions. An unsuccessful physical
release retains its cleanup owner. Explicit shutdown reports barrier/release
errors; defensive Drop retains the executor when completion cannot be established
instead of freeing device-visible memory. This is conservative fault containment,
not a guarantee of successful resource reclamation.

## Bounded output and scheduling

Waiting requests consume bounded request/input capacity but allocate executor
state only on successful admission. Rejected/deferred admission retains no
sequence resources. Physical memory budgeting remains the executor's responsibility.

Output has aggregate and per-request event limits. Work reserves credits before
submission, including terminal-event capacity. Completed results go to stable
mailboxes that outlive execution-slot reclamation. `pop_event_for(request)` drains
one client directly; `pop_event()` visits ready clients round-robin. Per-request
ordering is guaranteed; cross-request arrival ordering is not a public semantic.

Scheduling skips a blocked mailbox so peers with available model/output capacity
can progress. Aggregate saturation still backpressures work, and clients retaining
all available sequence capacity can prevent new admission. The current runtime
has no network-disconnect policy or state preemption.

Ready queues use stable internal slots and reused batch buffers. One batch is in
flight at a time. Decode receives priority with bounded admitted-prefill progress.
The scheduler does not know individual expert identities or attention algorithms.
Profiling, compilation, and expensive policy search remain outside this loop.

## Preparation and current limits

`Engine::with_defaults` derives compatible limits from a prepared executor;
`Engine::new` checks explicit limits strictly. The ordinary default constructor
must not reject a one-sequence executor merely because a generic default said eight.

`QwenCuda::load_gguf` prepares the existing CUDA text implementation, checks free
memory against weight/state/headroom budgets, and completes preparation before
returning. `ribn run` streams output through it; `ribn local` is the old numerical
comparison frontend. `ribn inspect` reads generic GGUF metadata without allocating
weights or initializing CUDA, and does not claim execution support from metadata.

The Qwen path remains experimental on the new runtime. Its kernel geometry,
state representation, full-context reservation, and greedy sampling remain those
of the existing implementation. Artifact-independent metadata does not establish
arbitrary-shape execution or support for another checkpoint format.

The new runtime still lacks a canonical compatibility/qualification manifest,
automatic variant selection, state paging/reuse, asynchronous wakeup integration,
multimodal task input, and an HTTP server. See the gap map rather than treating a
small trait as proof these capabilities exist.

## Project boundaries

CUDA Rust remains the NVIDIA kernel direction; target-specific vendor libraries
are allowed. The migration must preserve numerical and resource-ownership proof.
Future Metal/AMD implementations need real backend tests, not generic kernel
claims. Distributed inference may eventually coordinate tensor/expert/pipeline
execution within allocated resources. Fleet allocation and datacenter policy
remain external; Archon is not a required dependency.
