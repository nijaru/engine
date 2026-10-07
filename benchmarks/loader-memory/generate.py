"""Generate synthetic F32 BERT weights for host-loader measurements, not model parity."""

import argparse
import json
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
root = parser.parse_args().output
root.mkdir()
config = {
    "model_type": "bert",
    "hidden_act": "gelu",
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
}
(root / "config.json").write_text(json.dumps(config))
shards = [[] for _ in range(4)]


def add(name, shape, group):
    shards[group].append((name, shape))


add("embeddings.word_embeddings.weight", [30522, 768], 0)
add("embeddings.position_embeddings.weight", [512, 768], 0)
add("embeddings.token_type_embeddings.weight", [2, 768], 0)
add("embeddings.LayerNorm.weight", [768], 0)
add("embeddings.LayerNorm.bias", [768], 0)
for layer in range(12):
    prefix = f"encoder.layer.{layer}"
    group = 1 + layer // 4
    for path, shape in [
        ("attention.self.query", [768, 768]),
        ("attention.self.key", [768, 768]),
        ("attention.self.value", [768, 768]),
        ("attention.output.dense", [768, 768]),
        ("intermediate.dense", [3072, 768]),
        ("output.dense", [768, 3072]),
    ]:
        add(f"{prefix}.{path}.weight", shape, group)
        add(f"{prefix}.{path}.bias", [shape[0]], group)
    for path in ["attention.output.LayerNorm", "output.LayerNorm"]:
        add(f"{prefix}.{path}.weight", [768], group)
        add(f"{prefix}.{path}.bias", [768], group)
add("pooler.dense.weight", [768, 768], 3)
add("pooler.dense.bias", [768], 3)
weight_map = {}
parameter_id = 0
for i, tensors in enumerate(shards):
    filename = f"model-{i:05d}.safetensors"
    header = {}
    offset = 0
    for name, shape in tensors:
        n = math.prod(shape) * 4
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + n],
        }
        offset += n
        weight_map[name] = filename
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with (root / filename).open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for name, shape in tensors:
            count = math.prod(shape)
            value = struct.pack("<f", (parameter_id % 23 - 11) / 16)
            chunk = value * (1 << 18)
            while count:
                n = min(count, 1 << 18)
                f.write(chunk[: n * 4])
                count -= n
            parameter_id += 1
(root / "model.safetensors.index.json").write_text(
    json.dumps({"weight_map": weight_map})
)
print(root, "bytes", sum(p.stat().st_size for p in root.glob("*.safetensors")))
