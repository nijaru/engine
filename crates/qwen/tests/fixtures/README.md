# Hardware qualification fixtures

These fixtures feed `cargo test -p engine-qwen --features cuda --test cuda_runtime`,
which needs `RIBN_MODEL` and `RIBN_REFERENCE`:

```sh
RIBN_MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
RIBN_REFERENCE=$(pwd)/crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
cargo test -p engine-qwen --features cuda --test cuda_runtime \
  prepared_qwen_matches_reference_and_preserves_cancelled_peers \
  -- --ignored --exact --test-threads=1
```

`qwen38-code-fill4096-257.tokens` uses ribn's three-line format: artifact identity,
prompt token IDs, expected greedy continuation token IDs.

## Provenance

The expected tokens must never come from the implementation under test.

| Field | Value |
| --- | --- |
| Reference engine | llama.cpp `b81c99b479d4c24e5eeca10de99032ebd343ef8f` (2026-09-02), `llama-server` |
| Artifact | `Qwen3.8-27B-UD-Q4_K_M.gguf`, sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Prompt source | first 257 tokens of `benchmarks/incumbent-qwen38/workloads/code-fill4096.txt` |
| Tokenization | `POST /tokenize` with `{"add_special": false}` on the same artifact, so prompt IDs come from the artifact's own tokenizer rather than a reconstruction |
| Generation | `POST /completion` with `{"prompt": [ids], "n_predict": 32, "temperature": 0, "top_k": 1, "top_p": 1, "seed": 1, "ignore_eos": true, "return_tokens": true, "add_special": false}` |
| Semantics | no chat template, no BOS insertion, greedy, no stop filter, so every returned token is compared directly |

Both prompt and output are fixed token IDs, so the reference is independent of any
later tokenizer, template, or sampler change. Regenerate with a different artifact only
by repeating the steps above and recording the new values here; a fixture whose
identity line does not match `RIBN_MODEL` fails the harness's artifact check.

## Why the fixture holds eighteen of the thirty-two generated tokens

llama.cpp generated 32 tokens for this prompt. ribn reproduces the first 18 exactly at
concurrency 1 and flips one step at generated index 18, after which the two sequences
legitimately diverge because their contexts differ. The step is a genuine near-tie in
the reference itself: llama.cpp's top-2 logit gap there was 0.2242 nats
(`6249` at -0.7045, `592` at -0.9287) and ribn prefers `592`, the runner-up. A
differently ordered quantized GEMV and attention reduction can move a margin that
small, and the measurement says it does: across the steps before the sequences part,
ribn's log-probabilities sit 0.08-0.26 nats from llama.cpp's while agreeing on the
winning token, and at the flip step ribn's own margin is 0.0088 nats over the same two
tokens. The full per-step table is in `benchmarks/qwen-prefill-qualification.md`.

The committed reference is therefore the agreed prefix. The full llama.cpp
continuation, recorded so the truncation is never mistaken for a complete match:

```text
6488 283 1876 198 285 638 83841 470 283 21979 470 1358 727 2706 34071 5164 21761 25
6249 8 1411 21427 2561 25 198 262 4071 14048 264 6891 6249 1083
```

The same divergence appears with same-sequence prefill chunking disabled, and the
chunked path emits the identical token stream as the serial path, so this is a
pre-existing long-prompt numerical item rather than an execution-path difference. It is
not covered by the short-prompt fixture this harness used before, which matched 200
greedy steps because a five-token prompt gives drift much less room to accumulate.
