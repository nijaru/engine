# Runtime contract verification

## Host-only gates

The `ribn` crate has no third-party or CUDA dependencies. External fixture model
implementations exercise its real public trait with different private state
layouts. Qwen bridge tests inject a host backend into the actual translation and
lease code. These tests do not evaluate model numerics or execute GPU kernels.

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p engine-nvidia --features cuda --all-targets -- -D warnings
cargo clippy -p engine-qwen --features cuda --all-targets -- -D warnings
cargo clippy -p engine-server --features cuda --all-targets -- -D warnings
cargo test -p engine-qwen --features cuda
cargo test -p engine-server --features cuda
```

CUDA-feature compilation and the nonignored adapter tests run on a CPU CI host.
Device tests remain explicitly ignored. The redesign session ran these checks
with Rust 1.98.0; it did not have an NVIDIA device.

Coverage includes atomic batch commitment, delayed completion, in-flight
cancellation with peer progress, multi-token completion, live-policy changes,
queue/input limits, output credit reservation, admission rollback, deferred
admission, invalid output, release retries, partial submission and poll failures,
shutdown retries, and conservative retention after uncertain teardown.

## Synthetic CPU benchmark

```sh
cargo run -p ribn --release --example host_overhead
```

This reports median/p99 nanoseconds per warm scheduling iteration and median
cost per row at concurrency 1/8/32/128. Each run warms 128 iterations and measures
10,000. The mock immediately completes work and allocates its result vectors;
those costs are included. The event queue has room for both completed output
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
cargo run -p engine-server --features cuda -- run \
  --model /absolute/path/model.gguf --prompt 'Explain a mutex.' --max-tokens 128
```

`run` uses the new runtime and reports preparation/memory diagnostics on stderr.
`local` retains the previous correctness frontend. Both currently target the
existing Qwen text artifact path, not arbitrary GGUF architectures. Compare token
fixtures rather than raw stdout: `run` streams bytes without an added newline,
whereas `local` prints a finalized string. A token limit can truncate a UTF-8
code point in the raw streaming output.
