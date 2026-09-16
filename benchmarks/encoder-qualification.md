# BERT encoder device qualification

Date: 2026-09-15
Status: hardware-qualified at `f544f9c` for the fixture geometry and for the
pool-backed runtime wiring. Adversarial lifecycle qualification — delayed completion,
cancellation before and after enqueue, failed handoff and partial-enqueue retirement —
is roadmap slice 3c.

Result: the device encoder reproduces an independent Hugging Face `transformers`
reference for all four fixture cases, worst absolute deviation `4.77e-7` on hidden
states and `1.77e-8` on pooled output — fp32 accumulation-order agreement, not a
tolerance grant. The same build runs the encoder through `ribn-batch` under a
constrained shared pool and hands each result to its consumer device-resident.

## What this qualifies

`engine-bert` runs a BERT-style bidirectional encoder on CUDA: embedding summation,
mean-centred LayerNorm, exact-erf GELU, masked scaled-dot-product attention, dense
projections on cuBLAS, and a pooler head. The primitives live in
`crates/nvidia/src/encoder_ops.rs`; the model composition and parameter mapping live
in `crates/bert`.

This is the first non-decoder model path in the repository. It exists as the real
device work that the resource protocol's encoder/prepared-resource representation
must be determined from (roadmap slice 3), and as a second model family that
pressure-tests model-local integration.

## Oracle

The oracle is deliberately not another Ribn entrypoint. `crates/batch/tests/fixtures/
bert-tiny/generate.py` seeds PyTorch, builds a `BertModel`, and writes
`config.json`, `model.safetensors` and `reference.json`; the expected
`last_hidden_state` and `pooler_output` values come from Hugging Face
`transformers` 4.53.2 / torch 2.14.0. The host reference in
`crates/batch/tests/bert_architecture.rs` consumes the same fixture, so the device
path is checked against the same independent implementation rather than against the
host path.

Fixture geometry: `vocab_size=13`, `hidden_size=12`, `num_hidden_layers=2`,
`num_attention_heads=3`, `intermediate_size=20`, `type_vocab_size=3`,
`max_position_embeddings=16`, `hidden_act=gelu`, `layer_norm_eps=1e-12`, fp32.

Cases: one single-token request, one three-token request with real attention scores,
one four-token request with two token types, and one padded request whose mask
removes the last position as a key.

## Command

Run serially on an idle GPU, one model process at a time:

```sh
ssh desktop
cd ~/github/nijaru/engine
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
cargo test --release -p engine-bert --features cuda --offline \
  -- --ignored --test-threads=1 --nocapture
```

## Results

Measured 2026-09-15 on the idle RTX 4090 (driver 615.71.09), Rust 1.98.0, release
mode. Absolute deviation from the reference, one element at a time, tolerance
`2.0e-5`:

| Case | hidden elements | `last_hidden_state` max abs | `pooled_output` max abs |
| --- | ---: | ---: | ---: |
| `single_token` | 12 | 1.19e-7 | 7.45e-9 |
| `real_attention_scores` | 36 | 2.38e-7 | 1.77e-8 |
| `segment_types` | 48 | 2.38e-7 | 1.49e-8 |
| `padded_tail` | 48 | 4.77e-7 | 1.68e-8 |
| worst over cases | — | **4.77e-7** | **1.77e-8** |

Two further device tests pass: a submission reports its real completion state and
its device-byte envelope and can be read repeatedly after completion, and requests
outside the model's vocabulary/type/position range are refused before touching the
device with the encoder still usable afterwards.

## Runtime wiring

`crates/bert/tests/encoder_runtime.rs` runs the same encoder as a `ribn-batch`
executor. All three tests pass in the command above, in the same serialized run as
the parity cases:

- **The shared pool bounds execution.** With a pool sized to exactly two requests'
  envelopes and three queued requests, one step executes two and the next reports
  `Blocked(Pool { requested, available: 0 })`. The envelope the pool was charged for
  equals the storage that actually exists: `EncoderSubmission::device_bytes` counts
  the live buffers and is asserted against `CudaBertEncoder::request_bytes`, so a
  prediction that under-reports the device footprint fails here.
- **A device-resident result keeps its charge across dequeue.** Taking a completion
  out of the runtime leaves the pool charged until the result is dropped: a stalled
  consumer blocks a sibling, and releasing one result admits exactly one queued
  request. The consumer establishes completion before reading.
- **Permanent infeasibility is rejected while a peer progresses.** A sequence longer
  than the model's position embeddings is rejected as `SequenceTooLong`, and a
  sequence whose envelope exceeds the whole pool is rejected as
  `RetainedOutputTooLarge`; the healthy request queued behind each one still runs.

Host evidence for the same contract (constrained pool, deferred completion that must
be awaited before reading, charge surviving dequeue, a stalled consumer beside a
healthy peer) is in `crates/batch/tests/{capacity_readiness,device_result_handoff}.rs`,
which exercise the real `BatchRuntime` and `BytePool` against a fixture device.

## Limits

- One tiny fixture geometry. A production-size encoder must be re-qualified at its
  own width, depth and sequence length; nothing here claims support for any specific
  published checkpoint.
- fp32 only. No quantized, f16/bf16 or fused execution variant exists on this path.
- Attention runs one block per (query position, head) and is not a throughput
  implementation. This is correctness evidence, not a performance claim, and no
  serving comparison follows from it.
- The masked case covers key masking, not a fully masked query row; that row is
  defined to produce zeros rather than NaN, which the fixture does not exercise.
- Runtime wiring is exercised, not qualified under stress. Delayed completion with a
  genuinely lagging consumer, cancellation before and after enqueue, a failed
  handoff, and retirement after a partial enqueue are roadmap slice 3c; the failed
  submit path currently drains the stream best-effort instead of owning a retirement
  handshake.
- The whole request envelope stays charged until its result is dropped. Releasing
  completed temporary storage early (the activations after the last kernel) is not
  implemented, so peak pool demand is the request's full footprint rather than its
  settled footprint.
- Batching means several single-request submissions enqueued on one stream, not one
  padded device batch. A fused batch would need its own numerical and throughput
  qualification.
