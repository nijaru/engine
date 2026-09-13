# Qwen multi-token prefill qualification

Date: 2026-09-12
Status: hardware-qualified at `0496f79` on the pinned artifact; production serving still
runs serial prefill, and the candidate is not selected by serving.

Result: both parity gates pass and chunked prefill is 1.71x-1.79x serial at the
8-member cap with no decode regression, so the candidate is eligible for
backend-local serving integration. The measurement is the intermediate path only; it
is not a chunked GDN or multi-token attention implementation. See the qualification
result section at the end of this document.

This gate decides whether Ribn's experimental same-sequence Qwen prefill path is
safe and useful enough to consider for serving. It is deliberately separate from
the scheduler: the scheduler already emits prompt chunks, while this work changes
how the Qwen/CUDA backend executes one chunk.

The current candidate is an intermediate optimization. It batches prompt-token rows
through token-independent projection, normalization and FFN work, but keeps
convolution history, Gated DeltaNet recurrent-state updates, KV append and causal
attention ordered by token. A separate host-only test proves the true chunked GDN
triangular formulation against the direct recurrent definition; that algebra is the
oracle for a later scan/chunk CUDA kernel, not evidence that such a kernel is already
implemented.

## Preconditions

Use an authorized, idle RTX 4090. Do not evict or interrupt another GPU workload to
run this gate.

Record before every qualification session:

```sh
git rev-parse HEAD
nvidia-smi
sha256sum /home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
rustc --version
cargo --version
```

Keep driver/CUDA versions, clocks or power limits, other visible GPU processes, and
thermal state with the raw logs. Use one unchanged release build and one unchanged
model artifact for compared timings.

## 1. Correctness: hybrid-prefix diagnostic

This faster ignored test stages the first four language layers (three recurrent,
then one full-attention layer). It compares every prompt-row hidden state from a
four-token same-sequence chunk against serial batch-1 prefill, then executes one
additional continuation token through both persisted states.

```sh
cargo test -p engine-nvidia --features cuda --test cuda_reference \
  same_sequence_prefill_chunk_matches_batch1_hybrid_prefix \
  -- --ignored --exact --test-threads=1 --nocapture
```

Use this first because failures are cheaper to bisect than the full model. A failure
means the candidate is not eligible for serving selection regardless of benchmark
results.

## 2. Correctness: full 64-layer model

This ignored test stages the complete 64-layer Qwen text path, runs eight prompt
tokens through serial batch-1 and same-sequence chunk prefill, compares every final
layer hidden row, then executes a continuation token through both states. That last
step is important: matching prompt residuals alone is not enough if recurrent or KV
state was persisted incorrectly.

```sh
cargo test -p engine-nvidia --features cuda --test cuda_reference \
  same_sequence_prefill_chunk_matches_batch1_full_model \
  -- --ignored --exact --test-threads=1 --nocapture
```

Both ignored parity tests currently use the pinned local path
`/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf`. If the qualification
artifact moves, update the test path deliberately rather than silently substituting
a different checkpoint.

Passing this test qualifies the current eight-row candidate only for the tested
artifact, implementation and tolerance. It does not qualify arbitrary Qwen geometry,
other quantization formats, or a future true chunked GDN kernel.

## 3. Prefill timing: serial baseline versus chunk execution

The benchmark stages the model before starting the prefill timer, so the reported
prefill interval excludes the one-time checkpoint read/device upload. Decode remains
the same host-driven batch-1 path in both modes.

Run a serial baseline and chunk sizes 2, 4 and 8 at multiple prompt lengths. Keep the
output budget fixed so the benchmark remains within its 512-token test KV cache.

Example 257-token matrix:

```sh
export ENGINE_QWEN_GGUF=/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf

cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
  --tokens=64 --prompt-tokens=257
cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
  --tokens=64 --prompt-tokens=257 --prefill-chunk=2
cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
  --tokens=64 --prompt-tokens=257 --prefill-chunk=4
cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
  --tokens=64 --prompt-tokens=257 --prefill-chunk=8
```

Repeat the matrix at prompt lengths 65, 257 and 447. With `--tokens=64`, 447 is the
largest of these that fits the benchmark's 512-token state allocation. Run each cell
at least five times after one discarded warmup run and keep all raw output.

For each cell record median prefill wall time and spread. Also retain decode time as
a regression check even though the candidate does not intentionally change decode.
Do not compare staging time as part of the prefill result.

## Promotion rule

Do not wire the candidate into serving unless all of the following are true:

- both ignored parity gates pass on the pinned 4090/artifact;
- repeated chunk timings show a meaningful prefill improvement over serial execution
  on more than one prompt length and the improvement is larger than run-to-run noise;
- no material decode regression appears in the same runs;
- memory use remains safe on the qualification GPU;
- cancellation/error lifetime rules remain conservative when the path is integrated
  through the serving dispatcher.

If chunk size 8 wins only on one synthetic prompt or the effect is within timing
noise, keep the path experimental and continue directly toward a true chunked GDN
and multi-token attention implementation instead of adding serving complexity for a
weak result.

## After the intermediate gate

The next performance step should be evidence-driven. The host test
`gdn_chunk_reference.rs` already validates the unit-lower-triangular chunk transform
against token recurrence for nonzero initial state and multiple chunk sizes. If the
intermediate path demonstrates that multi-token prompt execution is valuable, use
that oracle to qualify a backend-specific chunked GDN kernel that scans recurrent
state across chunks rather than across every token.

Full-attention prefill should likewise move toward an actual multi-token attention
implementation rather than repeatedly invoking the decode attention kernel. These
are model/backend execution changes; they do not require inventing a new top-level
scheduler contract.

## Qualification result, 2026-09-12 (head `0496f79`)

Session preconditions, recorded before the run:

```text
git rev-parse HEAD      0496f792d22ae93c3db465bb781fe19afbf4f5e0
artifact sha256         322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482
rustc / cargo           1.98.0 / 1.98.0
driver / CUDA UMD       615.71.09 / 13.4
GPU                     RTX 4090, 24564 MiB, idle (only gnome-shell/Xwayland resident)
```

Across the 72 timing runs the GPU stayed at 45-57 C and 241-361 W with no reported
HW or thermal slowdown. Raw output is kept at
`/home/nick/ribn-prefill-gate-2026-09-12/`; benchmark output is gitignored by the
`benchmarks/results/` policy, so the summary below is the committed record.

### Correctness

Both ignored gates passed on the pinned artifact:

```text
same_sequence_prefill_chunk_matches_batch1_hybrid_prefix  ok  164.78 s
same_sequence_prefill_chunk_matches_batch1_full_model    ok  173.74 s
```

The hybrid gate compares every prompt row of a four-token chunk through three
recurrent layers plus one full-attention layer against serial batch-1 execution, then
runs a continuation token through both persisted states. The full gate does the same
across all 64 layers with an eight-token prompt. Passing qualifies this implementation
at this artifact and tolerance (5.0e-3), and nothing else.

### Prefill timing

Three prompt lengths, four modes, six runs each with rep 0 discarded as warmup, one
unchanged release build, `--tokens=64`. Mode order rotates per repetition so no mode
is confounded with thermal drift. Values are medians of the five measured runs in
seconds; the parenthesized number is speedup against that prompt length's serial
median.

| prompt tokens | serial | chunk 2 | chunk 4 | chunk 8 |
| --- | --- | --- | --- | --- |
| 65 | 2.786 | 2.772 (1.01x) | 1.900 (1.47x) | 1.556 (1.79x) |
| 257 | 11.573 | 11.460 (1.01x) | 7.931 (1.46x) | 6.549 (1.77x) |
| 447 | 21.134 | 20.844 (1.01x) | 14.681 (1.44x) | 12.350 (1.71x) |

Per-cell spread was 0.006-0.029 s, so every speedup above is two orders of magnitude
outside run-to-run noise. Decode medians were 2.921/3.215/3.505 s at 65/257/447 prompt
tokens and within 0.002 s of that in every chunked mode at the same prompt length, so
no decode regression appears. Chunk 2 is indistinguishable from serial; the win grows
with chunk size up to the 8-member cap, which is consistent with amortizing weight
reads across prompt rows.

Device memory during a chunk-8 447-token run peaked at 16221 MiB used of 24564 MiB
(7940 MiB free) including the full staged model, so the candidate does not create
memory pressure at this workload size.

### Decision

This is promotion case C: parity passes and the prefill win is substantial and
repeatable across all three prompt lengths, so the candidate is eligible for
backend-local serving integration with the scheduler contract unchanged. Independent
same-sequence lane sizing, backend subchunking, keeping the logits-producing prompt
token on the existing output path, and conservative asynchronous fault handling are
the integration conditions. Production serving was not changed in this session, so
serving still executes prompt tokens serially and this path stays opt-in through the
benchmark flag until that integration is qualified in turn.
