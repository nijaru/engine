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

Provenance, so the expected tokens are never produced by the implementation under
test:

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
