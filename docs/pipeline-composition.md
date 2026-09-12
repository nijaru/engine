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

The current pressure test validates a narrow version of this boundary. A batch
encoder produces `Arc`-owned prepared state. AR executor admission receives stable
`RequestId` separately from internal `SequenceId`, takes ownership of the state for
that request, and the test proves handoff order does not determine which request
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
encoder item becomes relevant. The coupled VLM pressure test therefore needs to run
at the AR scheduler/resource boundary rather than pretending the validated sequential
handoff solves this case.

## Do not choose an opaque request handle too early

A single opaque "prepared multimodal input" handle would be easy to add to the
current `TokenRequest`, but it may be too coarse. Efficient VLM execution can need
per-item readiness and cache lifetime because different encoder items become relevant
at different prompt positions.

Do not stabilize that handle yet. The whole-input sequential case has now passed;
the remaining pressure test is prompt-positioned multimodal encoder items interleaved
with AR prefill.

The resulting AR request/resource contract should expose only the minimum dependency
information scheduling actually needs. Raw media stays above the runtime in the
model processor/application layer.

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

## Validation targets

Completed sequential pressure tests:

- batch-encoder -> AR request-state handoff;
- handoff correlated by stable request identity rather than incidental FIFO order;
- in-process state retains the same allocation in the fixture rather than being
  serialized across the stage boundary;
- cancellation before admission reclaims producer-owned state;
- cancellation after admission reclaims executor-owned state through `release`;
- missing prepared state fails only the affected AR request.

Remaining before stabilizing composition contracts:

- use an actual encoder-decoder architecture/checkpoint to validate real prepared
  device-state and cross-attention semantics;
- pressure-test VLM-style prompt-positioned encoder dependencies with separate
  encoder compute/cache pressure;
- let that evidence determine whether AR scheduling needs explicit per-item
  dependency descriptors, a resource-manager interface, or another representation.

The goal is not a universal pipeline graph. The goal is to preserve direct fast
paths while giving genuinely heterogeneous models the coordination they require.
