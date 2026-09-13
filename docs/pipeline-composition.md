# Pipeline composition and coupled execution

Date: 2026-09-12
Status: design-validation note; boundaries remain provisional

Ribn should not turn every architectural component of a model into a top-level
runtime stage. A stage boundary is justified by execution lifecycle, scheduling,
resource ownership, placement, or failure/cancellation behavior—not merely because
a model diagram contains two boxes.

This distinction matters most for encoder + generation models.

## Sequential encoder-decoder

Some models have a naturally sequential boundary:

```text
input media / source tokens
        |
        v
batch/encoder runtime
        |
        | prepared encoder state
        v
AR decoder runtime
        |
        v
output tokens / text
```

An ASR- or translation-style encoder can often finish a whole input before decoding
begins. In that case the shallow orchestrator can own the dependency between the two
runtimes. If both stages are co-located, the prepared encoder state should remain in
device memory; a logical stage boundary must not imply host serialization or RPC.

The current pressure tests validate a narrow version of this boundary. A batch
encoder produces `Arc`-owned prepared state. AR executor admission receives stable
`RequestId` separately from internal `SequenceId`, takes ownership of the state for
that request, and the tests prove handoff order does not determine which request
receives which state.

Cancellation also establishes a useful ownership rule without a new generic cleanup
interface:

- before AR admission, prepared state is still owned by its producer/orchestrator,
  which reclaims it when the downstream request is cancelled;
- after admission, ownership has moved to executor sequence state and the existing
  executor `release` path reclaims it after cancellation/completion is established.

This is validation of the ownership transition, not a finished orchestrator API. A
real encoder-decoder model still needs to test device state, cross-attention layout,
async completion, failure propagation, and version compatibility.

## Coupled multimodal generation

A VLM or similar multimodal generator can be different. Encoder items may correspond
to placeholder spans inside the generation prompt. Encoder work may be scheduled,
cached, prefetched, or evicted independently while AR prefill advances through the
prompt.

Current serving systems demonstrate this coupling. vLLM V1 tracks multimodal feature
items on the request, maintains a separate encoder-output cache, and accounts for an
encoder-compute budget alongside token scheduling. TensorRT-LLM likewise exposes
item-level multimodal encoder scheduling and optional encoder prefetching. These
systems do not require every multimodal encoder to finish as a separate external
pipeline stage before AR execution begins.

References:

- https://docs.vllm.ai/en/latest/api/vllm/v1/core/encoder_cache_manager/
- https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- https://nvidia.github.io/TensorRT-LLM/llm-api/reference/MultimodalConfig.html

For Ribn this means a multimodal model may use a **coupled generation runtime** that
coordinates two resource domains:

```text
processed multimodal items
        |                     prompt token progress
        v                             |
encoder batching/cache               |
        |                             |
        +------- prepared features ---+
                      |
                      v
               AR model execution
```

The AR scheduler still must not understand images, audio, preprocessing libraries,
or backend tensor classes. It may, however, need model-prepared dependency metadata
such as feature identity, prompt span/position, readiness, and resource cost when
those facts affect whether a prompt range can execute.

The current AR `Engine` deliberately hides prompt-prefix progress from an external
orchestrator. That is desirable for ordinary staged execution, but it means a simple
wrapper around the existing engine cannot efficiently decide when a prompt-positioned
encoder item becomes relevant. Coupled VLM decisions therefore belong at a
scheduler/resource boundary rather than in an external stage wrapper.

## What the VLM pressure test proved

`crates/runtime/tests/multimodal_dependencies.rs` models only scheduler-visible facts,
not images or processor implementation details. Each prepared encoder dependency has
an item identity, prompt span, encoder-compute cost, encoder-cache cost, and cached
or uncached state. Token progress and encoder resources are budgeted independently.

The test establishes several behaviors:

- a future encoder item is not eagerly scheduled before the prompt range that needs
  it;
- an encoder item can be computed in the same scheduling iteration whose prompt
  chunk reaches and consumes it;
- cached encoder output can cross its placeholder span without spending encoder
  compute budget;
- insufficient encoder compute truncates prompt progress before the uncached item;
- insufficient encoder cache capacity is a distinct failure/pressure mode from
  insufficient encoder compute;
- multiple prompt-positioned items can progressively truncate an otherwise valid AR
  prefill chunk;
- starting directly at an unready dependency stalls token progress until the
  required encoder resource is available.

This is enough to reject two premature designs:

1. every encoder must finish as a top-level stage before AR prefill starts;
2. one opaque whole-request multimodal handle is necessarily sufficient for
   efficient scheduling.

It is **not** enough to stabilize a production dependency descriptor. In particular,
the test's compute/cache integer units are fixtures, not a proposed arbitrary
resource vector. An actual VLM integration should determine whether the production
seam is explicit per-item descriptors, a model/resource planner queried by the AR
scheduler, or another small cooperative interface.

### What the request contract can carry

`crates/runtime/tests/multimodal_admission.rs` asks the same questions through the
real `Engine` instead of a pure function, and the answers divide the list above.

Carried today:

- `Admission::Deferred` is a working per-request encoder gate: the engine retries the
  request on later steps and commits no prompt token while it waits, so "do not start
  before the encoder output exists" is expressible without a new interface;
- a backend can perform encoder work inside its own step, and encoder output published
  by anyone is reused rather than recomputed, across requests as well as across steps.

Not carried today:

- a prefill completion must advance *exactly* the chunk the engine chose, so a backend
  cannot return a shorter range and stop before an unavailable placeholder. Decode may
  advance partially; prefill may not. The remaining options are to do the work anyway,
  exceeding an internal encoder budget, or to fail the submission, which faults every
  request in that batch rather than only the one whose encoder is missing;
- encoder budgets therefore hold only when `SchedulePolicy::prefill_chunk_tokens`
  aligns with the prompt's encoder-item granularity. Two of the three admission tests
  pin this: one shows aligned chunking staying inside budget, the other shows an
  eight-token chunk spanning both items and overspending in a single step because the
  model had no way to refuse.

That is the concrete reason the incremental `prepare`-then-`enqueue` contract in
[the resource protocol](resource-protocol.md) exists: negotiation before a step is
fixed, rather than a policy chunk the backend must honor or fail on.

## Do not choose an opaque request handle too early

A single opaque "prepared multimodal input" handle would be easy to add to the
current `TokenRequest`, but it is too early to make that the common contract.
Efficient VLM execution can require per-item readiness and cache lifetime because
different encoder items become relevant at different prompt positions.

The whole-input sequential and prompt-positioned coupled cases have now both been
pressure-tested. The next step is not another synthetic abstraction: integrate an
actual VLM processor/model path and expose only the minimum scheduler/resource
information that implementation demonstrably needs. Raw media stays above the
runtime in the model processor/application layer.

## Identity and cache validity

Prepared encoder state is derived state. Reuse must account for the identities that
can change its meaning, including at least the relevant model/parameter/adaptor
version and processor/configuration identity. A hot weight update must not silently
reuse stale encoder outputs.

This is the same principle already used for continuation state: reusable state is
valid only for the model state that produced it.

## Orchestrator rule

The top-level orchestrator remains shallow:

- use it for genuine cross-runtime dependencies, cancellation, output ordering,
  placement and backpressure;
- keep tightly coupled scheduling inside a specialized/composite runtime when the
  model benefits from joint resource decisions;
- keep the single-stage path direct;
- do not introduce Worker/Executor/Stage wrapper layers merely to make all models
  fit one pipeline abstraction.

Omni and media-generation models may therefore mix both approaches. A model can have
several logical stages while one stage internally coordinates multiple tightly
coupled components.

## Validation status and next targets

Completed sequential pressure tests:

- batch-encoder -> AR request-state handoff;
- handoff correlated by stable request identity rather than incidental FIFO order;
- in-process state retains the same allocation in the fixture rather than being
  serialized across the stage boundary;
- cancellation before admission reclaims producer-owned state;
- cancellation after admission reclaims executor-owned state through `release`;
- missing prepared state fails only the affected AR request.

Completed coupled pressure tests:

- prompt-positioned encoder dependencies participate in AR prefill progress;
- encoder computation can occur just in time for the prompt span that consumes it;
- cached items bypass encoder compute but still represent cache residency;
- encoder compute and cache are independent scheduling constraints;
- later dependencies can shorten a chunk after earlier dependencies were satisfied.

Remaining before stabilizing composition contracts:

- use an actual encoder-decoder architecture/checkpoint to validate real prepared
  device-state and cross-attention semantics;
- integrate an actual VLM processor/model and let its real feature tensors, prompt
  positions, cache lifetime and device costs determine the production coupled
  scheduler/resource seam;
- validate version compatibility and async failure propagation for both staged and
  coupled derived state.

The goal is not a universal pipeline graph. The goal is to preserve direct fast
paths while giving genuinely heterogeneous models the coordination they require.
