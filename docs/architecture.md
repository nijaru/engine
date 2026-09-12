# Architecture

Ribn is a Rust-first inference runtime. Model execution, request scheduling,
state ownership, preparation, and inference-local distribution belong here.
Fleet provisioning and datacenter scheduling belong to external systems such as
Archon. Ribn must remain independently usable.

## Current migration boundary

`crates/runtime` exports the dependency-free `ribn` crate. It is the target for
new request/runtime work. `crates/qwen` implements its `PreparedModel` boundary
using the existing Qwen GGUF/CUDA executor. `engine-server run` exercises that
integration; `engine-server local` retains the previous serving path as an
explicit comparison oracle. The new CUDA integration is experimental and has
not yet been qualified on hardware.

`crates/core` still contains legacy state, plan, and serving types used by the
Qwen implementation and its reference tests. They are not the extension API for
new models. This coexistence is a migration stage, not two permanent runtimes.
See [the redesign decision](runtime-redesign.md) and [cutover gates](roadmap.md).

## Ownership

```text
CLI / eventual HTTP and local frontends
             |
   model-specific input preparation
             |
        ribn::Engine
  request slots / scheduling / events
             |
       PreparedModel
  admission / submit / poll / release / synchronize
             |
   concrete model implementation
  typed sequence state / model resources / execution
             |
  target-specific backend and kernels
```

The application composes a prepared model with an Engine. It does not supply
layer descriptions, KV geometry, CUDA streams, or tensor maps to the scheduler.
A prepared model binds architecture, artifact, backend, state representation,
execution implementation, and preparation limits. Shared model resources and
per-sequence resources remain separate inside that implementation.

## Runtime contracts

### Requests and sequences have distinct identities

`RequestId` identifies user-visible work. `SequenceId` identifies model-owned
continuation state. IDs are engine-issued and process-unique; neither is an
allocation pointer, persistent cache key, or proof of model compatibility.
Internal slots can be reused only after their terminal ownership is resolved.

The Engine retains prompt progress, committed prefix, output progress,
cancellation intent, and queue membership. Physical state stays in the prepared
model, with concrete types appropriate to that implementation. Another model
can use entirely different state without extending a core state-family enum.

### Prefix is a semantic boundary, not a physical layout

A committed prefix counts model inputs consumed. Generated output is separate:
final prefill can emit the first output token without consuming that token.
A decode completion can advance more than one input and return the corresponding
accepted outputs. Draft work and rejected tokens never appear in committed state.

A prefix alone cannot restore, fork, or migrate state. Those operations require
real retained contents, representation compatibility, and explicit completion
proof. Fresh admission creates prefix-zero state only. Mixed device/host state,
compressed formats, sharing, and bounded reconstruction are model implementation
choices, not a universal core tensor/state IR.

The existing Qwen bridge still uses its original F16 KV and F32 recurrent bundle,
fixed context reservation, and single-device storage. Its old semantic/physical
coupling has been isolated, not erased throughout the repository.

### Submission is not completion

`submit` may enqueue device work and return. The Engine validates the entire
returned batch against the saved sequence IDs, prefixes, and budgets before
committing any row or emitting output. Live scheduling-policy changes do not
invalidate an earlier submission's contract.

In-flight cancellation records intent and suppresses uncommitted output after
completion; it never releases live state or cancels peers implicitly. A submit,
poll, or malformed-completion error fences further submissions. Uncertain
physical resources remain owned until synchronization establishes completion.

Explicit `shutdown` reports synchronization and release errors and retains a
retry owner. Defensive Drop synchronizes; if safe teardown cannot be established,
it deliberately retains the model owner rather than freeing device-visible
resources. This trades a fault-path leak for avoiding premature destruction.
Applications should call shutdown explicitly rather than relying on Drop.

### Admission and output are bounded

Waiting requests consume bounded queue/input capacity but do not reserve model
state until admission. A deferred or rejected model admission must retain no
sequence resources. The prepared model enforces physical capacity; the Engine
enforces request and event limits.

In-flight work reserves output credits, including room for a terminal event.
Unrelated cancellations cannot consume those credits. Terminal events are
independent of physical release; a release failure retains its cleanup owner
and does not duplicate output. Delivery requires the application to keep driving
the Engine and draining events while it is alive.

### Scheduling is small and explicit

`BatchItem` contains sequence, prefill/decode kind, prefix, token budget, and
output budget. Encoders, attention algorithms, experts, and speculative methods
are not global scheduler phases.

The initial new runtime uses persistent slots, reused batch buffers, decode-first
selection, and a bounded number of decode-only steps before admitted prefill
gets a chunk. This is a step fairness bound, not a latency guarantee. One batch
is in flight at a time; internal backend streams/stages remain implementation
choices. Multi-batch overlap must earn its complexity in measurements.

## Preparation, compatibility, and precision

Preparation resolves immutable model/backend checks before request scheduling.
Device compilation, weight loading, and scratch preparation stay off the token
hot path. `QwenPrepared::load_gguf` observes free memory, budgets weight staging
against state reservations and explicit headroom, and completes preparation
before returning. Its memory report is an estimate/check, not immunity from
later allocation failures or other GPU users.

Numerical formats, kernel choices, and state layouts remain exact compatibility
dimensions. Changing layout or precision is not automatically lossless merely
because semantic state names match. Qualification must bind the actual artifact,
implementation, runtime/device capabilities, representation, graph mode,
speculation, and distributed layout. A display name, device ordinal, process ID,
or Rust TypeId is not a durable fingerprint.

The legacy qualification system remains in use only where integrated. The new
Qwen path is explicitly experimental; there is no new automatic variant selector
or persistent compatibility cache yet.

## Input and task boundaries

The new public request surface currently accepts encoded text-generation input.
Chat templates, thinking controls, BOS/EOS handling, and tokenization stay in the
frontend/model-input layer. Unsupported generation semantics fail before model
state allocation; the first Qwen adapter supports greedy sampling only.

This is not yet a universal multimodal, embedding, audio, or diffusion API.
Those tasks need explicit owned input/output contracts and tests before a stable
API promise. Do not smuggle non-text payloads through invented text tokens.
Attention changes alone should not require changes to the text-generation API;
genuinely different task semantics may require additive task interfaces.

## Backend and project boundaries

CUDA Rust remains the intended NVIDIA kernel foundation: cuTile and cuda-oxide
behind one coherent resource owner, with vendor libraries allowed. The new
request contract does not change that direction or claim its proof gates passed.
See [CUDA migration](cuda-rust-migration.md).

Future Metal, AMD, and distributed implementations use the same request and
ownership contracts while keeping hardware mechanisms specialized. Ribn does
not need a universal tensor compiler, dynamic Rust plugin ABI, shared KV-page
format, or required fleet manager to support that boundary.
