# Pipeline composition: sequential and coupled execution

These are host pressure tests, not implemented multimodal support or a production
orchestrator. [Target design](inference-engine-design.md) owns architecture;
[resource protocol](resource-protocol.md) owns device handoff and resource lifetimes.

A model component is not automatically a runtime stage. A stage boundary needs a
real scheduling, ownership, placement or failure boundary. Single-runtime models
should not traverse pass-through stage/worker/executor layers.

## Sequential encoder → decoder

`crates/batch/tests/sequential_encoder_decoder.rs` hands encoder-produced `Arc` state
to AR admission, keyed by `RequestId` rather than incidental FIFO order. Two handoffs
installed in reverse order still reach the correct decoder requests without copying
or serializing the allocation.

`RequestId` correlates request-scoped prepared input. `SequenceId` identifies the
continuation owner after admission; neither proves allocation lifetime.

- Before successful AR admission, the producer/composition owner retains retirement
  responsibility, including cancellation and failed handoff.
- After admission, sequence state owns the handoff and executor release reclaims it
  only after completion is established.
- Missing prepared state rejects that request, not its peers.

The fixture proves correlation and an ownership transition. A real encoder-decoder
model must still prove device completion dependencies, cross-attention layout,
version/processor compatibility, cancellation and shared-pool downstream headroom.
An `Arc` alone does not establish safe device reuse.

## Coupled multimodal generation

An encoder item can correspond to a placeholder span partway through an AR prompt.
Finishing every encoder as an external stage before prefill is therefore not the only
useful composition. A whole-request opaque handle may also hide when a feature becomes
relevant.

`crates/runtime/tests/multimodal_dependencies.rs` is a pure scheduling counterexample.
It models feature identity, prompt spans, readiness and separate encoder compute/cache
costs, not images or processors. It demonstrates:

- just-in-time encoder work at the prompt span that consumes it;
- cached features consuming residency but no encoder compute;
- unavailable compute shortening progress before an uncached item;
- cache pressure being distinct from compute pressure;
- later dependencies shortening a chunk after earlier ones were satisfied.

`multimodal_admission.rs` exercises the real AR engine. An authority-backed deferred
admission can wait for whole-request encoder readiness. A backend can do encoder
work within its step and reuse published features. Positive shortened prefill can
report less work than offered, within an aggregate submission budget, and only final
prefill samples output. Impossible indivisible work rejects at admission.

The former completion-time `Blocked` result was removed: it named no readiness source
and could immediately resubmit unchanged work. Partial completion is evidence of work
already done, **not preparation**. The current API cannot park an already-admitted row
at a temporarily unready leading dependency. This requires real pre-submit
negotiation, not a zero-progress success result or `yield_now` loop.

The fixture's integer cost units do not define a production resource vector. Raw
media and processing stay above token scheduling. A real VLM should determine the
minimum model-prepared dependency seam the scheduler needs.

## Remaining gate

Roadmap slice 4e requires a real second decoder and VLM/processor, followed by a
sequential encoder-decoder and small iterative model. Qualify independent numerical
references, derived-state compatibility, async failure propagation, cancellation and
constrained-pool progress before stabilizing composition interfaces. Keep both staged
and coupled execution available where the actual model justifies them.
