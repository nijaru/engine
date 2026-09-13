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

## 4. Correctness: consecutive chunks

The single-chunk gate proves one chunk against one serial prefix. Serving drives chunks
in sequence, so a second gate runs three consecutive eight-token chunks (24 prompt
tokens) through all 64 layers, compares every chunk's rows against the serial batch-1
hidden states for the same positions, then continues one token on both paths:

```sh
cargo test -p engine-nvidia --features cuda --test cuda_reference \
  same_sequence_multi_chunk_prefill_matches_batch1_full_model \
  -- --ignored --exact --test-threads=1 --nocapture
```

This covers what a single chunk cannot: KV rows and recurrent state carried across a
chunk boundary. A stale convolution history, a GDN matrix advanced out of prompt order,
or a position-dependent full-attention write would only diverge from the second chunk
on, and would leave a one-chunk gate green.

## 5. Correctness and effect through the serving seam

The integration is gated, not only the kernels:

```sh
cargo test -p engine-nvidia --features cuda --test cuda_reference \
  serves_chunked_prefill_matching_the_serial_path \
  -- --ignored --exact --test-threads=1 --nocapture
```

One 21-token prefill segment goes to two dispatchers that differ only in the prefill
lane: the chunked one runs two eight-token chunks, four serial tokens, then the sampling
token, while the serial one runs all 21 tokens serially. Both must sample the same
token and agree on the following eight decode steps. The prompt is the base prompt plus
the first sixteen tokens llama-server generated for it, so the expected continuation
comes from an independent engine. The test also asserts the lane is configured, because
a regression that silently dropped it would otherwise leave the test green while
measuring nothing.

Serving-level effect uses the serving benchmark, which drives the real scheduler and
runtime rather than the decoder directly, at a 257-token prompt and one unchanged
release build:

```sh
ENGINE_QWEN_GGUF=/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
target/release/examples/qwen_serving_bench \
  --concurrency=4 --tokens=32 --prompt-tokens=257 --print-tokens [--prefill-chunk=8]
```

The prompt is the same five tokens cycled to length, which makes this a prefill-cost
fixture rather than a realistic prompt distribution; the token streams it prints are
the parity check for the same run.

## 6. Where prefill time goes

Before optimizing further, measure. `nsys` over one 257-token prefill with one decode
step, release build, pinned artifact:

```sh
ENGINE_QWEN_GGUF=/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  -o /tmp/prefill target/release/examples/qwen_decode_bench \
  --prompt-fixture=crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
  --tokens=1 --prefill-chunk=8
nsys stats --report cuda_gpu_kern_sum /tmp/prefill.nsys-rep
```

| mode | prefill wall | GPU kernel time | kernel launches |
| --- | --- | --- | --- |
| chunk 8 | 6.582 s | 6.55 s | 69,512 |
| serial | 11.629 s | 11.44 s | 309,864 |

Both modes are GPU-bound; wall time is within 2% of kernel time, so launch overhead is
not the limiter and neither is host code. Composition of the chunk-8 prefill:

| family | GPU time | share |
| --- | --- | --- |
| quantized GEMV (`*_gemv_warp_batch`) | 5.41 s | 83% |
| attention (`attn_score_gqa`) | 0.68 s | 10% |
| recurrent/GDN kernels | 0.34 s | 5% |
| norms, rope, elementwise | ~0.12 s | 2% |

The GEMV share is the finding, and one kernel pair shows why it is larger than it
should be. `iq4_xs_gemv_warp_batch` decodes each weight element once per launch and
reuses it across all eight members, so a chunk-8 launch reads exactly the bytes a
batch-1 launch reads:

| kernel | launches | time per launch | weight bandwidth |
| --- | --- | --- | --- |
| `iq4_xs_gemv_warp` (batch 1) | 30,186 | 83 us | ~447 GB/s |
| `iq4_xs_gemv_warp_batch` (batch 8) | 3,744 | 479 us | ~95 GB/s |

Same bytes, 5.8x the time. Per member the batch kernel is therefore only 1.38x cheaper
(60 us against 83 us) although it reads one eighth of the weights per member, and its
throughput is about a tenth of the card's HBM peak. Switching kernel family does not fix
it: `--gemv=int-dot` completes the same chunked prefill in 5.346 s (1.23x) and the same
serial prefill in 10.851 s (1.07x).

The consequence for sequencing is that the two named next steps are worth about 15% of
prefill between them - attention 10% and the recurrent scan 5% - while a batched GEMV
that reached the batch-1 kernel's own bandwidth would cut roughly 2.8x off the whole
prefill (5.41 s to about 1.9 s) with no numerical change and no new kernels. Optimize
the batch GEMV's memory behavior first; the multi-token attention and chunked GDN work
remains worth doing, and is now second.

Not measured here: achieved occupancy and stall reasons, which need `ncu`. Profiling
counters are unavailable on this host (`ERR_NVGPUCTRPERM` without root), so the cause
inside the batch kernel is inferred from bandwidth and launch geometry rather than
observed. *Resolved: the batched-GEMV result section below names the cause (strided
scalar activation loads) and measures the fix; the ordering above is now satisfied, and
multi-token attention leads the remaining work at 27% of prefill kernel time.*

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

The next performance step should be evidence-driven, and section 6 is that evidence.
The host test `gdn_chunk_reference.rs` already validates the unit-lower-triangular chunk
transform against token recurrence for nonzero initial state and multiple chunk sizes,
so a backend-specific chunked GDN kernel that scans recurrent state across chunks rather
than across every token remains well supported and worth building. It is also worth
about 5% of prefill, while the batched quantized GEMV kernels hold 83% and run at a
fraction of the bandwidth the batch-1 kernels achieve on identical weight bytes. Fix the
GEMV memory behavior first; the GDN and attention kernels remain the next qualified
targets after it. *The GEMV fix landed in the batched-GEMV result section below, which
re-measures this composition: multi-token attention is now the largest single item and
the batch GEMV families the next.*

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

## Serving integration result, 2026-09-12 (heads `150bb62`..`4c22e11`)

Case C was carried out. Chunked prefill is now the backend's default prefill strategy
for this path, and nothing above the backend changed: a prefill segment is still one
segment with one outcome, the scheduler still asks for its 16-token prefill chunks,
and the lane is a private copy of the batch-1 executor's bindings that adds only
per-lane scratch. Whole leading chunks leave the serial path; the tail, and on a
sampling segment the final logits-producing token, stay serial. Lane size is
independent of `max_pending_rows` and of the decode lanes, so one long prompt chunks
even in a dispatcher prepared for two concurrent decodes - the serving-seam gate
asserts exactly that, because a regression silently tying the lane to request
concurrency would otherwise leave a differential test green while measuring nothing.

Session preconditions: the same artifact (sha256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`), driver 615.71.09
on an idle RTX 4090 (241 MiB used, 0% utilization, 43 C between stages), rustc/cargo
1.98.0. Raw logs and per-stage `nvidia-smi` records are in the session evidence
directory.

### Gates

| gate | test | head | result |
| --- | --- | --- | --- |
| single chunk, 4 layers | `same_sequence_prefill_chunk_matches_batch1_hybrid_prefix` | `0496f79` | ok, 164.78 s |
| single chunk, 64 layers | `same_sequence_prefill_chunk_matches_batch1_full_model` | `3183d44` | ok, 173.89 s |
| three consecutive chunks | `same_sequence_multi_chunk_prefill_matches_batch1_full_model` | `3183d44` | ok, 174.81 s |
| serving seam | `serves_chunked_prefill_matching_the_serial_path` | `4c22e11` | ok, 174.18 s |
| end-to-end runtime | `prepared_qwen_matches_reference_and_preserves_cancelled_peers` | `4c22e11` | ok, 1256.57 s |

The end-to-end gate is the only one that exercises the production loader, `Engine`,
scheduler, and live cancellation together: a 257-token prompt at concurrency 1/2/8/9
plus in-flight decode cancellation with peer progress at 2 and 8, against the
llama.cpp fixture in `crates/qwen/tests/fixtures`. Chunking was enabled by default for
that run, and the same fixture is what exposed the long-prompt item below. Two
construction faults in the new gates are part of the record only as fixes: the
three-chunk gate first used a KV capacity its own prompt outgrew, and the first
attempt at that fix raised the layer count instead of the token block.

### Serving effect

`qwen_serving_bench` driving the real scheduler and runtime, 257-token prompt, 32
output tokens, one unchanged release build (`754ab23`, whose serving code is identical
to `150bb62`):

| concurrency | prefill | TTFT | elapsed | ITL |
| --- | --- | --- | --- | --- |
| 1 | serial | 11.569 s | 13.113 s | 49.82 ms |
| 1 | chunk 8 | 6.595 s | 8.140 s | 49.83 ms |
| 4 | serial | 46.316 s | 50.716 s | 141.92 ms |
| 4 | chunk 8 | 26.369 s | 30.756 s | 141.52 ms |

Time to first token improves 1.75x at concurrency 1 and 1.76x at concurrency 4, with
inter-token latency unchanged, and the printed token streams are identical between
modes. TTFT at concurrency 4 is four times the concurrency-1 value in both modes,
because queued requests' prefills run one after another; chunking shortens each of
them rather than overlapping them. The serial TTFT of 11.569 s also reproduces the
decoder-level benchmark's 11.573 s serial prefill of the same prompt length through a
completely different seam, which is a useful cross-check on both benches.

### Long-prompt finding, and why it is not a chunking effect

The 257-token fixture diverges from llama.cpp at generated step 18, where llama.cpp's
own top-2 margin was 0.2242 nats and ribn prefers the runner-up token; the later
disagreements are that changed context, not additional flips. The item is pre-existing
and independent of this work: with chunking disabled the token stream is byte-identical
to the chunked stream, and both diverge at the same step. The earlier harness could not
see it because its five-token prompt gave drift far less room, which is why "200 greedy
tokens match" was never evidence about long prompts. It is recorded in the fixture's
provenance with the full reference continuation, and it is a numerical-fidelity item for
the long-prompt path, not a scheduling or ownership defect.

Margins were then measured rather than assumed, because two engines can pick the same
token for different reasons. `--logit-margins=N` reads the vocabulary logits back
through `CudaQwen35Decode::copy_logits` and prints each step's top five log-probabilities
and top-1/top-2 margin; `benchmarks/llama_reference_margins.py` prints the same rows from
llama.cpp's `n_probs` for the fixture's own prompt, fed as token IDs with `add_special`
disabled. Same 257-token prompt, serial prefill for the reference comparison:

| step | llama.cpp margin | ribn margin | margin error | worst shared top-5 error |
| --- | --- | --- | --- | --- |
| 0 | 0.8779 | 0.7967 | 0.0812 | 0.1064 |
| 11 | 3.6355 | 3.6820 | 0.0465 | 0.1055 |
| 12 | 0.1855 | 0.1655 | 0.0200 | 0.0910 |
| 14 | 0.3488 | 0.2044 | 0.1444 | 0.1245 |
| 17 | 2.9102 | 2.8642 | 0.0460 | 0.1040 |
| 18 | 0.2242 | 0.0088 | 0.2154 | 0.1953 |

Two things follow. First, the flip is not a discontinuity: ribn's log-probabilities
sit 0.08-0.26 nats from llama.cpp's throughout, so a step whose reference margin is
0.2242 is inside that band and either token is consistent with ribn's own
distribution. ribn's margin at that step, 0.0088 nats, is what a nearly-tied
preference looks like. Second, the deviation is slow and roughly flat in step index
rather than compounding, which is the signature of a systematic kernel-order difference
in the quantized paths rather than a state or position error.

That pass also produced a stronger statement about the integration than token equality:
**chunked and serial prefill emitted identical log-probability tables for all 20 steps**
(`diff` of the two runs is empty), so chunking changes which kernels execute, not what
they compute. The 20-step tables are in the session evidence directory as
`11-margins-reference.log` and `12-margins-ribn-{serial,chunk8}.log`. That property is
load-bearing, not incidental: it is why the chunked prefill gates below can compare
against the serial oracle at all. The batched-GEMV result section re-verifies it at a
later head with a different kernel-internal accumulation order; the tables recorded here
shifted by at most 2e-4 nats as a result, and the reference comparison and step-18 flip
above are unchanged.

### Margin probe procedure

```sh
python3 benchmarks/llama_reference_margins.py 20 > /tmp/llama-margins.txt

ENGINE_QWEN_GGUF=/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
target/release/examples/qwen_decode_bench \
  --prompt-fixture=crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
  --tokens=20 --logit-margins=20 [--prefill-chunk=8] > /tmp/ribn-margins.txt

diff <(grep '^margins' /tmp/llama-margins.txt) <(grep '^margins' /tmp/ribn-margins.txt)
```

### Decision

Chunked same-sequence prefill is selected by default through `QwenLoadOptions`, with
the scheduler contract, state ownership, cancellation semantics, and output path
unchanged. Still open in this area: true chunked GDN and multi-token attention (the
lane batches projections and feed-forward work per position while recurrence still
advances token by token), the long-prompt fidelity item above, and a serving-relevant
prompt-length sweep with a realistic prompt distribution rather than the repeated
timing fixture.

## Batched GEMV result, 2026-09-13 (head `f68e345`)

Section 6 put 83% of chunked prefill GPU time in the batched quantized-GEMV kernels
and left the reason inferred from bandwidth and launch geometry. The reason was the
activation access, not the weight read. The kernels already decoded each weight
element once per launch and reused it across all eight members, but each lane read
its eight activations one scalar at a time: a warp load spanned 32 different 32-byte
sectors and used 4 bytes of each, so a 1 KiB activation block cost eight L1
wavefronts where one suffices. The weight decode does not scale with the member
count while the per-member activation loads and FMAs do, which is why the batched
kernel landed at about a tenth of HBM peak and only 1.38x cheaper per member than
eight batch-1 launches.

### The change

Raw output - the `int_dot_bench` before/after tables, the nsys report and its kernel
summary, both margin tables, the serving streams, the two gate logs, and per-stage
preconditions (same artifact sha256 as above, driver 615.71.09, RTX 4090) - is in the
session evidence directory `/home/nick/ribn-gemv-activation-2026-09-13/`.

Each lane now owns four consecutive activations in the low half of a quantization
block (elements `4*lane`) and four in the high half (elements `128 + 4*lane`). Both
runs are 16-byte aligned, so one member's activations are two `float4` loads per lane
and a warp load covers 512 contiguous bytes. The same lane mapping turns the packed
nibbles of those four elements into one word load per half, so the decode-once
property survives: the K-quant families whose block stride is not a multiple of four
(110 and 210 bytes) read that word as two aligned halfwords, because a 32-bit load
inside such a block faults on the odd rows.

The batch-1 warp kernels take the same mapping and the same decode expressions. They
had the same activation access pattern, and sharing the accumulation order - rather
than only the layout - is what keeps chunked prefill bit-identical to serial prefill
instead of merely within a tolerance. The first attempt changed only the batched
kernels, and the three-chunk gate then measured 6.2e-3 against its 5.0e-3 tolerance:
a valid but different summation order through 64 layers. Restoring one shared order
removed the question instead of widening the gate.

`decode_f16` is now `cvt.f32.f16`. Every binary16 value is exactly representable in
binary32, so it is bit-identical to the sign/exponent expansion it replaces while
costing one instruction instead of a per-block shift loop. The 32-element block
families notice it most: one scale covers 32 elements there against 256 in the
K-quants, so the same decode is amortized over eight times fewer elements.

### Kernel-level effect

`crates/nvidia/examples/int_dot_bench --iters=50`, synthetic weights at representative
Qwen shapes, weight bytes counted once per launch:

| family | shape | batch 8 before | batch 8 after | batch 1 before | batch 1 after |
| --- | --- | --- | --- | --- | --- |
| IQ4_XS | 5120x5120 | 82.0 | 379.5 | 419.6 | 719.1 |
| IQ4_XS | 5120x17408 | 85.3 | 363.1 | 494.7 | 965.4 |
| IQ4_XS | 17408x5120 | 83.6 | 427.3 | 469.8 | 864.3 |
| Q4_K | 5120x5120 | 88.5 | 276.1 | 423.3 | 1027.4 |
| Q4_K | 5120x17408 | 87.2 | 433.3 | 517.3 | 1454.1 |
| Q4_K | 17408x5120 | 88.7 | 296.2 | 474.5 | 1353.8 |
| Q5_K | 5120x5120 | 105.6 | 323.3 | 477.5 | 975.0 |
| Q5_K | 5120x17408 | 107.8 | 504.1 | 574.0 | 1303.2 |
| Q5_K | 17408x5120 | 105.6 | 351.7 | 529.1 | 1178.5 |
| Q6_K | 5120x5120 | 117.3 | 403.9 | 578.5 | 1157.6 |
| Q6_K | 5120x17408 | 123.0 | 597.1 | 678.1 | 1537.7 |
| Q6_K | 17408x5120 | 67.2 | 418.4 | 639.1 | 1417.0 |
| Q8_0 | 5120x5120 | 25.2 | 63.1 | 44.7 | 224.4 |
| Q8_0 | 5120x17408 | 28.0 | 87.4 | 34.7 | 104.7 |
| Q8_0 | 17408x5120 | 20.8 | 49.0 | 26.6 | 83.4 |

Values are weight GB/s. The batch-1 rows show the same activation fix on the path
that decode and serial prefill use; the batch-8 rows are the ones the chunked lane
runs. Both columns include the `cvt.f32.f16` change, whose largest single effect is
in the 32-element block families (Q8_0 batch 1, 5120x5120, 5.0x); the K-quant gains
in both columns come from the activation remapping.

### Prefill, decode, and serving effect

`qwen_decode_bench` and `qwen_serving_bench`, pinned artifact, 257-token fixture, one
release build. Before values are the head `b5475b0` measurements recorded above.

| measurement | before | after | speedup |
| --- | --- | --- | --- |
| chunk-8 prefill (5 runs) | 6.553-6.567 s | 2.576-2.588 s | 2.54x |
| serial prefill (3 runs) | 11.587 s | 7.417-7.427 s | 1.56x |
| decode 20 tokens (same runs) | 1.054-1.133 s | 0.667-0.668 s | 1.60x |
| `--gemv=int-dot` chunk-8 prefill | 5.346 s | 4.478 s | 1.19x |

The integer-dot variant is no longer the faster family: the float warp path is now
1.7x faster than it on the same prefill, so the earlier 1.23x advantage in its favor
is inverted and nothing needs the quantize-then-dot round trip on this workload. Its
own 1.19x improvement is the float fallback path it uses for `Q3_K`/`IQ4_NL`/`IQ3_S`
plus the serial tail and sampling token, not its integer kernels, which do not call
`decode_f16`.

Through the serving scheduler and runtime at a 257-token prompt and 32 output tokens:

| concurrency | mode | prefill | TTFT | elapsed | ITL |
| --- | --- | --- | --- | --- | --- |
| 1 | serial | 7.430 s | 7.430 s | 8.470 s | 33.54 ms |
| 1 | chunk 8 | 2.581 s | 2.588 s | 3.627 s | 33.52 ms |
| 4 | serial | 29.653 s | 29.654 s | 31.715 s | 66.49 ms |
| 4 | chunk 8 | 10.301 s | 10.302 s | 12.363 s | 66.47 ms |

Chunking is now worth 2.87x of TTFT at concurrency 1 and 2.88x at concurrency 4,
against 1.75x and 1.76x before this change, and inter-token latency improves rather
than merely holding: 49.8 ms to 33.5 ms at concurrency 1 and 141.7 ms to 66.5 ms at
concurrency 4. The printed token streams at concurrency 2 are identical between serial
and chunked prefill (`diff` of the `tokens[...]` lines is empty), which is the
serving-seam check that this path still computes what it computed before.

### Where prefill time goes now

`nsys` over the same chunk-8 prefill. GPU kernel time is 2.56 s of a 2.605 s wall, so
the workload is still GPU-bound. Shares are of kernel time.

| kernel | before | after | speedup |
| --- | --- | --- | --- |
| `attn_score_gqa` | 682.5 ms | 683.2 ms | 1.00x |
| `iq4_xs_gemv_warp_batch` | 1794.5 ms | 433.1 ms | 4.14x |
| `q5_k_gemv_warp_batch` | 1476.7 ms | 410.6 ms | 3.60x |
| `q4_k_gemv_warp_batch` | 1287.4 ms | 300.2 ms | 4.29x |
| `gdn_state_update` | 276.2 ms | 277.8 ms | 1.00x |
| `q8_0_gemv_warp_batch` | 291.1 ms | 91.5 ms | 3.18x |
| `q6_k_gemv_warp_batch` | 132.1 ms | 38.9 ms | 3.40x |
| `q3_k_gemv_warp_batch` | 127.4 ms | 30.1 ms | 4.23x |
| `iq3_s_gemv_warp_batch` | 123.4 ms | 23.6 ms | 5.23x |
| `iq4_nl_gemv_warp_batch` | 100.3 ms | 40.3 ms | 2.49x |

The eight batched GEMV families fall from 5.33 s to 1.37 s together (3.9x). Multi-token
attention is now the largest single item at 26.7%, then the three large GEMV families
at 44.7%, then the recurrent scan at 10.9%. That ordering is the new short list:
`attn_score_gqa` is still the decode-shaped kernel invoked per token, and the remaining
GEMV time is activation traffic that more rows per warp would amortize.

### Numerics

Serial and chunked prefill emitted identical 20-step log-probability tables at this
head (`diff` of the two runs is empty), so the bit-identical property recorded in the
serving integration result survived the kernel rewrite. Both tables shifted by at most
2e-4 nats against the tables recorded at `0496f79` - for example step 14 moves from
0.2044 to 0.2040 nats and step 18 from 0.0088 to 0.0087 - with the same chosen tokens
and the same top-5 ranking at every step. The reference comparison above is unaffected:
ribn still sits 0.08-0.26 nats from llama.cpp and still flips the same near-tie of
0.2242 nats at step 18.

### Decision

The fix is a kernel-internal change: the scheduler contract, state ownership,
cancellation semantics, backend lane sizing, and the output path are untouched, and
the logits are unchanged in every way the gates can see. Chunked prefill stays the
default prefill strategy for this path. Next in this area are true multi-token
attention (now the largest single item), a chunked GDN scan, and the long-prompt
fidelity item and serving prompt-length sweep that remain open above.
