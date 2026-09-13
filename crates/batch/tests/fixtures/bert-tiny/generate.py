#!/usr/bin/env python3
"""Generate the independently-referenced BERT fixture used by `bert_architecture.rs`.

The committed fixture exists because an in-code fixture can only be checked
against expectations derived from the same implementation. Here the weights are
random but seeded, and every expected value comes from Hugging Face
`transformers`, which is an independent implementation of the same equations.

Run from the repository root:

```sh
uv run --with torch --with transformers python \
  crates/batch/tests/fixtures/bert-tiny/generate.py
```

This writes `config.json`, `model.safetensors`, and `reference.json` next to
this file. Re-running with different library versions changes the weights and
the reference together, so regenerate both or neither.
"""

import json
import pathlib

import torch
import transformers
from transformers import BertConfig, BertModel

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20_260_912

# Deliberately small but not degenerate: no zero projection matrices, nonzero
# position and token-type embeddings, two layers so the residual path is
# exercised more than once, and non-square feed-forward widths.
CONFIG = BertConfig(
    architectures=["BertForPreTraining"],
    vocab_size=13,
    hidden_size=12,
    num_hidden_layers=2,
    num_attention_heads=3,
    intermediate_size=20,
    max_position_embeddings=16,
    type_vocab_size=3,
    hidden_act="gelu",
    layer_norm_eps=1.0e-12,
    position_embedding_type="absolute",
)

CASES = [
    {
        "name": "single_token",
        "token_ids": [5],
        "token_type_ids": [0],
        "attention_mask": [1],
    },
    {
        "name": "real_attention_scores",
        "token_ids": [2, 7, 11],
        "token_type_ids": [0, 0, 0],
        "attention_mask": [1, 1, 1],
    },
    {
        "name": "segment_types",
        "token_ids": [4, 1, 9, 3],
        "token_type_ids": [0, 0, 1, 1],
        "attention_mask": [1, 1, 1, 1],
    },
    {
        "name": "padded_tail",
        "token_ids": [4, 1, 9, 3],
        "token_type_ids": [0, 0, 1, 2],
        "attention_mask": [1, 1, 1, 0],
    },
]


def main() -> None:
    torch.manual_seed(SEED)
    model = BertModel(CONFIG, add_pooling_layer=True)
    model.eval()
    model.save_pretrained(HERE, safe_serialization=True)

    cases = []
    with torch.no_grad():
        for case in CASES:
            output = model(
                input_ids=torch.tensor([case["token_ids"]], dtype=torch.long),
                token_type_ids=torch.tensor([case["token_type_ids"]], dtype=torch.long),
                attention_mask=torch.tensor([case["attention_mask"]], dtype=torch.long),
            )
            last_hidden_state = output.last_hidden_state[0].to(torch.float32)
            pooled_output = output.pooler_output[0].to(torch.float32)
            cases.append(
                {
                    **case,
                    "last_hidden_state": last_hidden_state.flatten().tolist(),
                    "pooled_output": pooled_output.tolist(),
                }
            )

    reference = {
        "generator": f"transformers {transformers.__version__} / torch {torch.__version__}",
        "seed": SEED,
        "config": {
            "vocab_size": CONFIG.vocab_size,
            "hidden_size": CONFIG.hidden_size,
            "num_hidden_layers": CONFIG.num_hidden_layers,
            "num_attention_heads": CONFIG.num_attention_heads,
            "intermediate_size": CONFIG.intermediate_size,
        },
        "cases": cases,
    }
    (HERE / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"wrote {HERE}/reference.json with {len(cases)} cases")


if __name__ == "__main__":
    main()
