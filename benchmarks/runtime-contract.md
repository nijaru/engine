# Runtime contract verification

## Host-only gates

The `ribn` crate has no third-party or CUDA dependencies. External fixture model
implementations exercise its real public trait with different private state
layouts. Qwen bridge tests inject a host backend into the actual translation and
lease code. These tests do not evaluate model numerics or execute GPU kernels.

```sh
python3 tools/check-boundaries.py
cargo test -p engine-qwen --no-default-features --test model_config
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p engine-nvidia --features cuda --all-targets -- -D warnings
cargo clippy -p engine-qwen --features cuda --all-targets -- -D warnings
cargo clippy -p ribn-cli --features cuda --all-targets -- -D warnings
cargo test -p engine-qwen --features cuda
cargo test -p ribn-cli --features cuda
```

CUDA-feature compilation and the nonignored adapter tests run on a CPU CI host.
Device tests remain explicitly ignored. The redesign session ran these checks
with Rust 1.98.0; it did not have an NVIDIA device.

Coverage includes atomic batch commitment, delayed completion, in-flight
cancellation with peer progress, multi-token completion, live-policy changes,
queue/input limits, output credit reservation, admission rollback, deferred
admission, invalid output, release retries, partial submission and poll failures,
shutdown retries, and conservative retention after uncertain teardown. New tests also cover
per-request mailbox isolation, draining after execution-slot reuse, bounded ready
list membership, source-format-independent model geometry, and small-executor defaults.

`crates/runtime/tests/progress_negotiation.rs` pins positive partial completion:
shortened prefill commits exactly the reported range, output belongs only to final
prefill, and decode output matches advancement. A two-credit output pool tests refunds
when an offered final-prefill range is shortened. Zero-prefill success is rejected
with a failing-before regression. Mixed malformed rows commit no healthy peer output.
`crates/runtime/tests/multimodal_admission.rs` checks shared submission budgets and
admission-time rejection of impossible items. It deliberately cannot express temporary
mid-request waiting; no completion-time `Blocked` API remains.

`crates/runtime/tests/abandoned_requests.rs` separates execution and mailbox
lifetimes: an execution slot can be reclaimed before its terminal event is consumed.
`Engine::discard` cancels and suppresses present/future delivery, even after that
slot is gone, while retaining in-flight retirement ownership. Tests cover repeated
discard without draining, healthy peers and in-flight cancellation. Output unit tests
pin reservation retention, ready-list unlinking and discarded terminal publication
when a peer holds all buffer capacity. Request-scoped callers use `pop_event_for`;
the aggregate drain intentionally includes other clients' events.

## Shared text frontend

`ribn-text` reuses the low-level runtime for raw prompt, chat-message, and token-ID
input. Since the owned text facade landed it is a thin layer over `ribn::driver`:
bounded preprocessing, incremental decoding and ordered batching above owned request
streams. Its behavior is host-qualified; only `TextOwner::load` (Qwen GGUF/CUDA
assembly) needs a device.

CLI behavior uses the same frontend: default text input is one user chat message,
`--raw` bypasses the chat template, and prompt/file/piped-stdin input share the same
generation implementation. GPU numerical qualification remains the separate hardware
gate below.

`TextResponse.text` and the `generate_batch` collect helper hold caller-collected
results and are deliberately outside buffered-application accounting. `TextBatch` is
the bounded incremental interface. CLI file/stdin ingestion is still a whole read and
is not claimed as bounded.

### Runtime-owned discard qualification (2026-09-14)

Revision `4477a0e` removes the text facade's discard/retry list. Batch collection
relinquishes all submitted requests on both success and error; stream engine/decoder
errors relinquish delivery immediately. Physical retirement remains runtime-owned.

- Host workspace tests, formatting, boundaries, default and CUDA-feature all-target
  clippy passed with actual zero exit statuses.
- At that revision, `discard_survives_a_blocked_in_flight_completion` failed before
  the cancellation branch fix and passed after it. The old branch erased cancellation
  intent. Removal of blocked outcomes later replaced this with partial-completion
  cancellation coverage.
- CUDA text lifecycle: **4/4, 77.32 s**, serialized on the RTX 4090 with the pinned
  Qwen artifact. Log: `desktop:~/ribn-lifecycle-2026-09-14/01-runtime-discard.log`.
- Other newly added discard tests exercise a new API, not failing-before tests against
  an API that previously existed. Host fixture tests do not exercise text decoding.
- Full CUDA kernel/reference and exact-token gates were not rerun for this slice;
  kernel arithmetic and the Qwen executor were unchanged. This is not a new numerical
  qualification claim. Concurrent handles and event-driven readiness remain undone.

### Positive completion and combined gates (2026-09-14)

Revision `14d0290` removes runtime `StepOutcome`/`StepStatus::blocked`. Qwen and all
runtime adapters return `StepCompletion` directly. Successful prefill advancement
must be positive; final-prefill output is exactly one token, intermediate output is
empty, and decode output equals advancement. Whole-batch validation borrows rows
without allocating/cloning a row plan; commit moves rows and retains batch capacity.
This is a source-level allocation removal, not an allocation-free runtime or measured
model-speedup claim.

The zero-prefill regression failed with the old predicate (exit 101) and passed with
positive advancement required. The partial-prefill test uses a two-event global pool
to require refund of unused final-prefill credits. Discard retirement tests also cover
failed release with mailbox reuse and uncertain completion through shutdown.

All required host checks passed with actual zero exit codes. Serialized device gates
against the pinned Qwen artifact on the RTX 4090:

| Gate | Result | Time | Log under `desktop:~/ribn-lifecycle-2026-09-14/` |
| --- | --- | --- | --- |
| Text lifecycle | 4/4, exit 0 | 76.91 s | `02-completion-text.log` |
| Exact-token runtime | 1/1, exit 0 | 174.22 s | `03-completion-runtime.log` |
| CUDA reference | 61/61, exit 0 | 408.13 s | `04-completion-reference.log` |

`completion-revision.txt` pins the tested commit; `completion-exits.txt` records each
process exit. Kernel arithmetic was unchanged. These gates do not qualify concurrent
handles, readiness, a real VLM, or the still-opt-in GDN scan.

### Owned token driver host gate (`7005fad`)

The first driver increment adds owned token streams, not a concurrent text/model API.
The 19 tests in `crates/runtime/src/driver/tests.rs` cover independent callers, stalled
consumers, retained terminal permits, stop-vector capacity, invalid configuration,
enqueue rejection, cancellation under saturated admission, cancellation without a
device batch, dropped submission/next futures, failed sync/release, shutdown retry,
cancelled shutdown futures, owner drop and worker panic.

Mutation checks observed exit 101, then pass with the mechanism restored:

- `abandoned_in_flight_requests_keep_their_admission_charge_until_retirement`: remove
  the runtime's private retention guard; a discarded route returns its permit while
  its sequence still retains input/options storage.
- `output_credit_return_between_check_and_park_is_not_lost`: omit consumption's wakeup;
  the test pauses the worker after its condition check and before sleep, returns a
  credit, then stalls at its three-second deadline (poll fallback is 60 seconds).
- `queued_submission_and_shutdown_acknowledgements_are_dropped_on_worker_panic`: omit
  exit's explicit queue drain; queued acknowledgement senders survive disconnection
  and the awaiting client hits its three-second deadline.

Required host checks, including CUDA-feature clippy, exited 0. The 19 driver tests
passed 100 consecutive suite runs. These tests exercise the actual driver, not a
separate scheduling simulation. No model arithmetic changed.
[Matched direct/driver measurements](runtime-alignment/README.md#owned-token-driver-7005fad)
record the observed host cost and raw CSV, not inference throughput.

**Device gate passed (2026-09-14, `dc511bd` + later):** desktop was reachable and idle
(RTX 4090, driver 615.71.09, GPU 0% / 33 MiB before the run). The checkout was pulled to
the tested revision and the pinned artifact verified by SHA-256 before running. Both
runtime tests passed serially, exit 0: **2/2 in 199.11 s**, including
`owned_driver_preserves_reference_with_stalled_and_abandoned_peers`. Private log:
`desktop:/tmp/gate1.log`. This qualifies the owned driver for the tested device path.

Superseded blocker (kept for history): desktop qualification and checkout synchronization
could not run because Tailscale was stopped and `ssh desktop` failed hostname resolution.
`owned_driver_preserves_reference_with_stalled_and_abandoned_peers` in the existing
Qwen CUDA runtime test is compiled but unrun. Do not infer GPU qualification from host
success. Once desktop is idle/reachable, run both runtime gates serialized:

```sh
RIBN_MODEL=/path/to/pinned/model.gguf \
RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
cargo test --release -p engine-qwen --features cuda --test cuda_runtime --locked \
  -- --ignored --test-threads=1
```

Text preprocessing/decoder injection, a bounded concurrent text facade and ordered
incremental offline batches are now implemented and host-qualified (see the owned text
facade gate below). Current CUDA text lifecycle evidence still applies to the earlier
direct path, not to the owned driver or the facade built on it.

### Owned text facade host gate (2026-09-14)

`crates/text/tests/facade.rs` drives the real `GgufProcessor`, a real GGUF tokenizer
built from synthetic byte-level metadata, the real token driver and the real facade
over a scripted fixture `GenerationExecutor`. Only the device is substituted, so the
preprocessing, delivery, batching and shutdown paths under test are production ones.

Twenty-one tests pass with no failures across consecutive suite runs:

- concurrent callers on cloned handles, plus the async submission path;
- a decode failure settling exactly one request while its peer completes;
- invalid UTF-8 reported with its offending token;
- a terminal flush replacing an incomplete trailing code point exactly once;
- cancellation returning a `Cancelled` terminal for its own request while the owner
  keeps serving (buffered-event preservation is proven by the driver tests, not here);
- oversized prompt/message/prompt-token inputs rejected before admission, with the
  same handle still serving afterwards (permit refund on the failure path);
- ordered per-item batch outcomes with a rejected input between healthy ones, and a
  decode failure inside a batch settling only that item;
- shared-handle overload: with every permit retained by stalled consumers, a third
  request and each batch item report `DriverError::Overloaded` for themselves, and
  the permits return once the consumers release;
- a pull-counting iterator proving batch lookahead stays inside the window;
- dropping a batch abandoning delivery without stranding the owner;
- shutdown closing admission while reporting success;
- zero preprocessing workers rejected at assembly;
- a stalled consumer not blocking a peer request;
- the decoded-payload bound rejecting one oversized token for its own request only;
- the rendered-prompt bound during template expansion, with the raw input inside its
  own bound;
- owner failure preserving delivered events, inventing no terminal, and reporting one
  error after which the stream stays exhausted;
- a failed text shutdown reported through its owner, with the same owner retrying and
  succeeding;
- a preprocessing pool whose only worker panics reporting failures to its caller and
  to later submissions instead of queueing them forever;
- permit refund under a two-permit pool where three rejected requests would exhaust
  the pool if any leaked;
- every `drain` assertion requiring exactly one terminal event, so a duplicated or
  missing terminal cannot pass unnoticed.

Required checks at this revision, each with captured exit 0:

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

**Device gate passed (2026-09-14):** the migrated lifecycle tests ran serially on the
RTX 4090 with the pinned artifact, exit 0: **5/5 in 95.83 s**
(`an_invalid_token_fails_only_its_own_batch_member`,
`an_unpreparable_batch_input_settles_only_its_own_item`,
`cancellation_is_request_local`, `dropping_a_stream_does_not_poison_a_later_batch`,
`repeated_abandonment_stays_bounded`). Private log: `desktop:/tmp/gate2.log`.

A further device test, `repeated_requests_are_deterministic_and_coherent`, covers what
those gates could not: four identical requests to one owner must reproduce identical
text (greedy sampling has no legitimate reason to change), which catches cross-request
delivery, shared state or a stale continuation. It passed, **1/1 in 19.52 s**, and the
example ran four concurrent callers, an abandoned stream and explicit shutdown
successfully. Private log: `desktop:/tmp/det2.log`.

Observed model behavior, **not** a pipeline defect: with greedy sampling and only
`eos_token_id` in the stop list, this artifact can emit chat special markers
(`<|endoftext|>`, `<|im_start|>`) as ordinary text and run to the token limit instead of
stopping at the turn marker. Repeated suffix prompts ("... Caller 3.") also degenerate
into a repeated-token loop. Both are prompt/policy behavior: identical requests stayed
deterministic across indices, and the same prompt at request index 0 and request index 6
produced identical coherent output. An earlier read of this as index-dependent
corruption was wrong. See the roadmap's chat stop-policy item.

Superseded status (kept for history): this layer's device gate was unrun. `TextOwner::load` and the Qwen executor are unchanged
arithmetic; this gate still needs a reachable device:

```sh
ENGINE_QWEN_GGUF=/absolute/path/model.gguf \
cargo test --release -p ribn-text --features cuda --test text_lifecycle --locked \
  -- --ignored --test-threads=1
```

`crates/text/examples/concurrent_text.rs` exercises concurrent callers, an abandoned
live stream and explicit shutdown against a loaded model. It ran successfully on the
RTX 4090 (four callers, exit 0). It is a usage example and a diagnostic, not a
qualification gate.

Matched direct-versus-handle frontend overhead on the real GPU path is still unmeasured;
the only measured host comparison remains the token-driver benchmark above.


## Synthetic CPU benchmark

```sh
cargo run -p ribn --release --example host_overhead
```

This reports median/p99 nanoseconds per warm scheduling iteration and median
cost per row at concurrency 1/8/32/128. Each run warms 128 iterations and measures
10,000. The mock immediately completes work and allocates its result vectors;
those costs are included. The aggregate event bound has room for both completed output
and the next batch's reserved credits. Admission, preparation, and teardown are
outside the samples.

It is not inference throughput, a GPU benchmark, or an old/new speed comparison.
Do not turn inverse iteration time into advertised model tokens/second. Run
multiple samples on a controlled CPU and record the revision, compiler, CPU,
load, and configuration. Allocation instrumentation and old/new matched traces
remain required before an allocation-free or performance-improvement claim.

## Hardware qualification

Only run on an authorized, idle qualification GPU. Follow the existing device
ownership checks in the CUDA probe/benchmark scripts. Do not interrupt another
GPU process to make this test pass.

Obtain a token fixture from the pinned llama.cpp reference or the independently
qualified old path, with the exact artifact and greedy/input semantics. The
fixture has exactly three lines:

```text
gguf:sha256:<64 lowercase hexadecimal digits of the complete GGUF>
<prompt token IDs separated by spaces>
<expected generated token IDs separated by spaces>
```

Use real token IDs and hash, not the placeholders. At least four output tokens
are required by the harness; use diverse 256/1024-token fixtures for the cutover
gate. Keep the reference engine revision, command/settings, tokenizer/template
semantics, and artifact metadata beside each fixture. The candidate implementation
must not produce its own expected output.

```sh
RIBN_MODEL=/absolute/path/model.gguf \
RIBN_REFERENCE=/absolute/path/reference.tokens \
cargo test -p engine-qwen --features cuda --test cuda_runtime \
  prepared_qwen_matches_reference_and_preserves_cancelled_peers \
  -- --ignored --exact --test-threads=1
```

The harness verifies artifact identity and exact tokens at concurrency 1/2/8/9,
then in-flight decode cancellation with peer progress at 2/8. It uses no stop
filter so reference tokens, including any stop ID, are compared directly. It
checks terminal ownership and shutdown. This is a starting harness, not all
cutover evidence: add mixed inputs, constrained memory, repeated admissions,
real cancellation/fallback combinations, and fault-path device evidence as
specified in the roadmap.

## Experimental CLI

```sh
cargo run -p ribn-cli --features cuda -- run \
  --model /absolute/path/model.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

`run` uses the new runtime and reports preparation/memory diagnostics on stderr.
`local` retains the previous correctness frontend. Both currently target the
existing Qwen text artifact path, not arbitrary GGUF architectures. Compare token
fixtures rather than raw stdout: `run` streams bytes without an added newline,
whereas `local` prints a finalized string. A token limit that cuts a UTF-8 code point
now ends the stream with one replacement character, because the facade flushes an
incomplete trailing code point at its terminal event.

## Ground-up alignment cost and isolation

The current mailbox implementation reserves both global and per-request event
capacity. Request-specific draining does not need an unbounded frontend event
buffer. Aggregate draining is round-robin across clients, preserving each client's
order rather than a global arrival order. Stalled-consumer and mailbox/slot reuse
tests run in the ordinary host suite.

[Host comparison](runtime-alignment/README.md) records the additional synthetic
CPU cost relative to the original global-queue runtime. Stable internal mailbox
slots avoid repeated hashing, but the result is not an inference speedup claim.
