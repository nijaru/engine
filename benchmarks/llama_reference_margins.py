#!/usr/bin/env python3
"""Record llama.cpp's greedy margins for a ribn reference fixture's prompt.

Token agreement cannot separate a matching distribution from a different one that
happens to pick the same winner, so long-prompt parity work needs margins as well as
tokens. This prints one line per generation step in the same shape the decode bench's
`--logit-margins` prints, so the two can be diffed directly:

```sh
python3 benchmarks/llama_reference_margins.py 20 > /tmp/llama-margins.txt

ENGINE_QWEN_GGUF=/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
  --prompt-fixture=crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens \
  --tokens=20 --logit-margins=20 > /tmp/ribn-margins.txt
```

The reference is an external engine, so the prompt is fed as token IDs with
`add_special` disabled, matching what ribn consumes. Requiring `llama-server` keeps
this out of the test harness deliberately: it is a measurement tool for the
qualification procedure, not a gate that runs in CI.
"""

import json
import pathlib
import subprocess
import sys
import time
import urllib.request

MODEL = "/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
FIXTURE = "crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens"
PORT = 8099


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fixture = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else FIXTURE)
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL
    prompt = [int(token) for token in fixture.read_text().splitlines()[1].split()]

    server = subprocess.Popen(
        [
            "llama-server", "-m", model, "-c", "1024", "-ngl", "99",
            "--host", "127.0.0.1", "--port", str(PORT), "--temp", "0", "--seed", "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("llama-server never became healthy")

        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": steps,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 1,
                "ignore_eos": True,
                "n_probs": 5,
                "cache_prompt": False,
                "add_special": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.load(response)
    finally:
        server.terminate()
        server.wait(timeout=60)

    for index, entry in enumerate(body["completion_probabilities"]):
        top = entry["top_logprobs"]
        row = " ".join(f"{t['id']}:{t['logprob']:.4f}" for t in top[:5])
        gap = top[0]["logprob"] - top[1]["logprob"]
        print(f"margins[{index}] chosen={top[0]['id']} gap={gap:.4f} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
