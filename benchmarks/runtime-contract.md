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

`crates/runtime/tests/progress_negotiation.rs` pins the completion contract: a
shortened prefill range commits exactly what was reported, sampled output belongs only
to the row that finished the prompt, a decode row may not report an empty successful
step or a token count that disagrees with its advancement, a blocked row must name its
own sequence, and a blocked row is reported and stays runnable rather than failing its
peers. `crates/runtime/tests/multimodal_admission.rs` exercises the same negotiation
against encoder-item budgets, including a chunk that spans two items completing
without chunk alignment.

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
input. Its synchronous streaming and offline batch paths are covered by workspace
and CUDA-feature compilation/tests. Terminal events include explicit prompt and
completion token accounting. The high-level stream currently borrows one model
mutably; it is not evidence of a concurrent server driver or HTTP compatibility.

CLI behavior uses the same frontend: default text input is one user chat message,
`--raw` bypasses the chat template, and prompt/file/piped-stdin input share the
same generation implementation. GPU numerical qualification remains the separate
hardware gate below.

Request ownership in that frontend is device-qualified separately, because
`TextModel::load` needs a Qwen CUDA executor and no host fixture constructs one:

```sh
ENGINE_QWEN_GGUF=/absolute/path/model.gguf \
cargo test --release -p ribn-text --features cuda --test text_lifecycle \
  -- --ignored --test-threads=1
```

It covers a stream dropped mid-flight followed by `generate_batch` (which
previously panicked, because the abandoned request's terminal event reached a
batch that indexed its own request map with it), six abandonments followed by an
ordinary request (undrained output can exhaust shared delivery capacity), an unpreparable batch input
failing before anything is submitted, and an invalid token failing only its own
batch member. Run it serially: each test loads the full 27B artifact, so parallel
processes contend for device memory. (The earlier parallel attempts failed on
descriptors first, at 1024 open files; that loader cost is fixed and recorded in
`docs/execution-foundation.md`.)

### Runtime-owned discard qualification (2026-09-14)

Revision `4477a0e` removes the text facade's discard/retry list. Batch collection
relinquishes all submitted requests on both success and error; stream engine/decoder
errors relinquish delivery immediately. Physical retirement remains runtime-owned.

- Host workspace tests, formatting, boundaries, default and CUDA-feature all-target
  clippy passed with actual zero exit statuses.
- `discard_survives_a_blocked_in_flight_completion` failed before the cancellation
  branch fix and passed after it. The old branch erased cancellation intent.
- CUDA text lifecycle: **4/4, 77.32 s**, serialized on the RTX 4090 with the pinned
  Qwen artifact. Log: `desktop:~/ribn-lifecycle-2026-09-14/01-runtime-discard.log`.
- Other newly added discard tests exercise a new API, not failing-before tests against
  an API that previously existed. Host fixture tests do not exercise text decoding.
- Full CUDA kernel/reference and exact-token gates were not rerun for this slice;
  kernel arithmetic and the Qwen executor were unchanged. This is not a new numerical
  qualification claim. Concurrent handles and event-driven readiness remain undone.

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
whereas `local` prints a finalized string. A token limit can truncate a UTF-8
code point in the raw streaming output.

## Ground-up alignment cost and isolation

The current mailbox implementation reserves both global and per-request event
capacity. Request-specific draining does not need an unbounded frontend event
buffer. Aggregate draining is round-robin across clients, preserving each client's
order rather than a global arrival order. Stalled-consumer and mailbox/slot reuse
tests run in the ordinary host suite.

[Host comparison](runtime-alignment/README.md) records the additional synthetic
CPU cost relative to the original global-queue runtime. Stable internal mailbox
slots avoid repeated hashing, but the result is not an inference speedup claim.
