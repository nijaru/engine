# Qwen multi-token prefill qualification

Date: 2026-09-12
Status: hardware gate pending; no GPU correctness or performance claim yet

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
