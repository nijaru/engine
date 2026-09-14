# Ribn agent guidance

Ribn (pronounced “ribbon”) is a Rust-first, **server-first general inference engine**.
Qwen GGUF/CUDA and the RTX 4090 are qualification workloads, not the product boundary.
Training is a future execution system; preserve useful lower-level reuse without
putting training policy into inference.

## Authority and working reference

- Work on `main` unless asked otherwise. All v0 APIs are unstable: replace flawed
  contracts and remove obsolete paths rather than add compatibility shims.
- At session start read `README.md`, `docs/inference-engine-design.md`,
  `docs/resource-protocol.md`, `docs/roadmap.md` and `docs/architecture.md`; then inspect
  the relevant source/tests. The first three design documents own target architecture,
  execution contracts and sequencing respectively.
- `docs/execution-foundation.md` and `docs/pipeline-composition.md` retain experimental
  evidence and limitations. Their provisional types are not mandatory architecture.
  Historical docs and private notes do not override current design owners.
- Resolve observable ownership, failure scope, cancellation, bounds, readiness and
  shutdown before implementing a dependent architectural slice. Record open decisions
  in the roadmap. A counterexample may justify changing the design; update the owning
  contract and tests rather than silently weaken it.
- Choose local data structures, helper names and implementation details within those
  contracts. Do not invent a universal framework to avoid making concrete decisions.
- Keep commits focused and the tree green. Preserve unrelated work and device artifacts.

## Architecture constraints

- Keep one mutable execution owner, bounded application submission and owned requests.
  CLI/HTTP/Python/Rust reuse model behavior, not separate execution loops. Single-model
  use does not require IPC, a stage graph or pass-through worker layers.
- `crates/runtime` (`ribn`) is the AR token runtime, not the universal inference API.
  Keep specialized scheduling for materially different execution regimes.
- Keep raw media, preprocessing and protocol types above token scheduling. A real VLM
  may require coupled encoder/AR scheduling rather than a separate encoder stage.
- Ordinary architecture additions should change model/processor/backend, registration
  and tests. Central enums/factories are fine; family branches throughout unrelated
  lifecycle/routing code are not. A second real decoder and real VLM must prove this.
- Keep architecture semantics separate from artifacts. GGUF and SafeTensors adapters
  expose format data; HF package resolution does not choose model semantics. Models
  own parameter interpretation and backend materialization.
- Physical layouts, kernels and device execution remain backend-specific. Do not build
  a compiler, plugin ABI, universal tensor layer or operator registry without concrete
  model/backend evidence. Preparation-time specialization is preferable to repeated
  hot-path support lookup where appropriate.
- Keep logical parameters separate from encoding, quantization, placement and storage.
  A version label is not executable ownership. Derived state must remain compatible
  with the coherent snapshot that produced it.
- Future training may share artifact/device/storage/completion/collective mechanisms.
  Gradients, autograd, activation retention and optimizer scheduling belong outside
  inference. Shared semantics do not require shared physical materializations.
- Logical model topology is separate from deployment topology. External systems own
  fleet allocation; Ribn owns execution within granted resources.

## Execution invariants

- `RequestId` identifies runtime requests; `SequenceId` identifies continuation owners.
  Do not collapse them or use identity as proof of allocation lifetime.
- Validate whole submissions before logical commitment; reserve output before launch.
  Successful progress is positive. Completed ranges are not pre-submit reservations.
- Cancellation is intent, not device completion. Failed/partial enqueue and failed
  cleanup retain an executor-owned retirement/quarantine path. Never release memory
  the device may still access.
- Discard ownership belongs with the runtime mailbox, including after an execution
  slot is reclaimed. Do not introduce frontend orphan-event or cancellation retry lists.
- Resource waiting must have a real readiness/reactivation protocol, including lost
  wakeup prevention and cancellation. `yield_now` is not parking. Permanently infeasible
  work needs request-local rejection, not endless retries.
- One accounting authority grants reservations per shared pool. Charges follow live
  allocations through handoff/dequeue until safe reuse. Bound aggregate submissions
  and prove downstream progress; count bounds alone are not byte bounds.
- Continuation is not necessarily KV. Reusable hybrid prefixes require all necessary
  attention/recurrent/other components to agree at the boundary.
- Current separate queues, full-context reservation and one in-flight batch are
  prototype mechanics, not permanent architecture. Change them with correctness and
  workload evidence, not merely to imitate another engine.

## Qualification and performance

Read `.agents/skills/model-integration/SKILL.md` for model integration, support review
or optimized-variant promotion. Read `.agents/skills/execution-contracts/SKILL.md` when
changing lifecycle, preparation, resource ownership or concurrent application access.

- Compilation and host tests are not GPU evidence. Qualify optimized variants over
  their entire automatic selection scope; preserve a qualified fallback.
- Numerical acceptance is independent of observed failure. Check shape and finiteness;
  bit equality applies when arithmetic preservation is intended. Reordered algorithms
  require independently justified metrics and tolerances. Compare common histories.
- GDN chunk scan remains opt-in while its full-model gate fails. Do not call it harmless
  or widen tolerances based on existing reference error.
- Do not repeat measured losers without new evidence: `MAX_BATCH_ROWS=4`, shared IQ4
  codebook, pre-elimination `decayed_keys`. See benchmark evidence for conditions.
- Serialize full-model device tests on `ssh desktop`; preserve its untracked
  `cuda-rust-probe/simt/*`. Profiler counters currently require unavailable privilege;
  do not infer measured occupancy/stall causes from speculation.
- Benchmark server workloads with mixed lengths/arrivals, memory pressure and latency
  objectives. One prompt's prefixes are not representative throughput evidence.

## Verification

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

Capture actual exit codes; pipes must not mask failures. The text facade is CUDA-gated,
so default host builds alone miss it. Device commands/evidence live in
`benchmarks/runtime-contract.md`; kernel-language qualification lives in
`docs/cuda-rust-migration.md`. Report failed or unrun checks explicitly.
