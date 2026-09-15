---
name: execution-contracts
description: Use when changing Ribn request lifecycle, concurrent handles, readiness, resource preparation or device handoff ownership; not for kernel-only tuning or ordinary model parameter mappings.
---

# Execution contract changes

Read `../../../docs/inference-engine-design.md`, `../../../docs/resource-protocol.md`
and the active gate in `../../../docs/roadmap.md`. Inspect the actual owner and its
callers. Historical experiments are counterexamples, not implementation mandates.

## Settle the contract before dependent code

- Identify every owner from input preparation through admission, submission, completion,
  output consumption/discard and retirement. Include failed partial enqueue.
- Define request-local versus execution-owner errors. An impossible request is not a
  device fault; uncertain physical state is not ordinary backpressure.
- Define count and byte bounds, reservation authority and when charges release.
  Include retained consumer allocations, not only queue occupancy.
- Check a payload bound where the payload is retained, not only where it is consumed.
  A check inside the consumer bounds whatever survived queuing, so aggregate retention
  becomes caller concurrency times the envelope rather than the declared bound.
- A queue owner that can lose its last worker must refuse new work and release queued
  work once that happens. A pool that keeps accepting jobs nothing will serve turns a
  processor panic into a caller that waits forever.
- Define wakeup registration/recheck, cancellation under saturation, no-batch readiness
  and shutdown. A repeated poll or `yield_now` is not a notification protocol.
- Put changed observable semantics in the existing contract owner before implementing
  dependents. Put unresolved choices/exit evidence in the roadmap, not a new plan file.
- If a real implementation disproves a proposed interface, change or remove it; do not
  add a compatibility layer solely to preserve unstable v0 APIs.

## Exercise adversarial transitions

Test partial positive progress, malformed mixed rows without logical commitment,
output-credit refunds, stalled and healthy consumers, abandonment before admission,
during execution and after terminal publication, decode errors and batch errors.
For concurrent owners test lost-wakeup interleavings, queue saturation, cancellation,
worker failure and shutdown, including a worker dying while peers are still queued.
For device handoffs test delayed writes/reads, failed handoff, attempted reuse and
retained charges under constrained pools.

Prefer host injection of the real implementation over contract-only mock facades.
Use an independent model/device reference for numerical behavior. State which tests
are genuine failing-before regressions and which demonstrate new behavior only.

## Verify and record

Run the roadmap checks, including CUDA-feature clippy for the gated facade. Capture
actual exit codes. Run affected device tests serialized; host tests are not device
qualification. Inspect the diff for new allocations, clones, lock/await lifetimes,
ignored errors and competing cleanup owners. Measure direct/handle overhead and
representative workload effects when introducing performance-sensitive machinery.

Update implementation status only after its exit evidence exists. Leave future
questions explicitly open rather than filling them with speculative public traits.
