#!/usr/bin/env python3
"""Matched-incumbent serving benchmark: drive llama-server's OpenAI-compatible
endpoint with the same request shape as the engine's serving sweep (prompt
"The capital of France is", 32 output tokens, greedy, all requests admitted
concurrently) and report aggregate throughput, TTFT, and ITL per concurrency.

Method-matched to benchmarks/run-qwen-serving-sweep.sh semantics: every
request is admitted at t=0; TTFT is the wall time to the first token per
request; ITL is the mean per-request inter-token interval; aggregate
throughput is total tokens / wall time from admission to last token.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = "The capital of France is"


def one_request(base_url: str, tokens: int, seed: int, index: int) -> dict:
    payload = json.dumps(
        {
            "model": "qwen",
            "prompt": PROMPT,
            "max_tokens": tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    token_times = []
    previous = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            chunk = line[len(b"data: "):].strip()
            if chunk == b"[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            content = event.get("content") or ""
            if content:
                now = time.perf_counter()
                if first_token is None:
                    first_token = now
                if previous is not None:
                    token_times.append(now - previous)
                previous = now
    finished = time.perf_counter()
    return {
        "index": index,
        "started": started,
        "first_token": first_token,
        "finished": finished,
        "itls": token_times,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    # Warm the endpoint with one throwaway request so model load and CUDA
    # graph/warmup effects do not pollute the measured run.
    one_request(args.url, 1, 12345, -1)

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.url, args.tokens, 42 + i, i)
            for i in range(args.concurrency)
        ]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - wall_start

    total_tokens = args.tokens * args.concurrency
    ttfts = [r["first_token"] - wall_start for r in results]
    itls = [delta for r in results for delta in r["itls"]]
    itl_mean = sum(itls) / len(itls) * 1000.0 if itls else 0.0
    itl_max = max(itls) * 1000.0 if itls else 0.0
    ttft_mean = sum(ttfts) / len(ttfts) if ttfts else 0.0
    print(f"concurrency: {args.concurrency}")
    print(f"aggregate throughput: {total_tokens / wall:.2f} tok/s")
    print(f"observed TTFT: mean {ttft_mean:.3f} s, max {max(ttfts):.3f} s")
    print(f"observed ITL: mean {itl_mean:.3f} ms, max {itl_max:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
