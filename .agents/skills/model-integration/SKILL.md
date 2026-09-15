---
name: model-integration
description: Use when adding a Ribn model architecture or processor, reviewing a model-support PR, or qualifying an optimized execution variant for automatic selection; not for ordinary refactors or unrelated documentation edits.
---

# Model integration and qualification

## Establish the change boundary

Read `../../../docs/roadmap.md` for the model-local integration acceptance gate and
current priorities. Read `../../../docs/inference-engine-design.md` and
`../../../docs/pipeline-composition.md` when the integration changes execution or
composition. If resource admission or handoff changes, also read
`../../../docs/resource-protocol.md` and `../execution-contracts/SKILL.md`.
The protocol owns lifecycle rules; exact preparation/readiness types require the
real implementation and roadmap gate, not transcription of a speculative sketch.

1. Identify whether this is an existing-architecture checkpoint, a new architecture
   using existing mechanisms, a new state/resource mechanism, or a new execution
   regime. State the supported artifact, operations, shapes, precision and devices.
2. Inspect the actual loader, processor, executor and tests; do not infer support
   from configuration enums, a registry entry or documentation alone.
3. Keep model semantics above artifact parsing. Keep raw media and model-family
   branches out of unrelated scheduling, cancellation, output routing and protocols.
   A central registration edit is fine. Explain each shared-contract change by a
   concrete model requirement; do not introduce a plugin ABI or generic graph solely
   to avoid editing registration.
4. Reuse qualified backend components where appropriate. Allow physical layouts and
   kernels to specialize. Preserve the direct single-runtime path; distinguish a
   genuinely sequential handoff from coupled encoder/AR scheduling.

## Review ownership before performance

- Trace request ownership through input errors, admission, abandonment, completion
  and retries. Exercise a dropped stream followed by a later request or batch, and a
  batch whose input fails to prepare: it settles only its own item while admitted
  peers stay owned. Cleanup must not leave orphan events or free in-flight resources.
- Verify count and byte bounds with mixed successes/rejections and stalled consumers.
  For shared device pools, reservations follow live allocations beyond result dequeue.
- Verify preparation can negotiate feasible work and ordinary backpressure without
  faulting unrelated requests. Check reservation rollback before submission and
  completion/quarantine ownership after partial submission.
- For handoffs, test delayed producer completion, consumer failure/cancellation and
  attempted reuse. Reference counting alone is not proof of device completion.
- Pin executable model identity; test derived-state compatibility when weights,
  adapters, processor configuration or physical representation change.

## Qualify the actual execution scope

1. Pin artifact identity, reference implementation revision, inputs, precision,
   hardware and execution options. Use an independent reference, not only a second
   entrypoint into the same implementation.
2. Check shapes and finiteness before reducing errors. Use bit equality when the
   transformation promises identical arithmetic; use justified absolute/relative
   bounds for reordered algorithms. State the metric precisely: a ratio of maxima
   is not a per-element relative-error bound.
3. Compare common input histories: teacher-force the same continuation or stop
   numerical aggregation after histories diverge. Token agreement alone is not
   distributional equivalence. Existing baseline error grants no extra error budget.
4. Test tails, chunk boundaries, longer contexts and continued execution. Check
   persistent continuation components as well as outputs. Diagnose failures using
   captured real inputs and higher-precision references; do not raise tolerances
   merely above the failing observation. An early failure leaves later work untested.
5. Exercise the intended user/runtime path. Explicitly identify legacy benchmarks
   rather than treating their results as evidence for a replacement runtime.
6. Measure matched baselines across the shapes/encodings/devices selected by the
   variant. Include mixed lengths/arrivals and memory pressure for serving claims;
   one prompt's prefixes are only a bounded timing sweep. Report repetitions and
   variability. Label bottleneck explanations as hypotheses unless discriminated by
   counters or experiments. Preserve a qualified fallback outside the measured scope.

## Deliver reviewable support

Run repository verification and the relevant device gates. Report unrun checks and
remaining unsupported scope. Record reproducible evidence in the established
benchmark documents and update roadmap status only when its exit evidence exists.
The PR summary should identify model-local changes, justified core changes, artifact
and operation coverage, numerical evidence, lifecycle checks and performance trade-offs.
