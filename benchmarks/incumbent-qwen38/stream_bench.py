#!/usr/bin/env python3
import json
import pathlib
import sys
import time
import urllib.request

base = sys.argv[1].rstrip('/')
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
workload_paths = sys.argv[4:] or ["chat.txt", "code.txt", "reason.txt"]
model = __import__("os").environ.get("MODEL_NAME", "qwen3.8:27b-awq")

for path in workload_paths:
    prompt = pathlib.Path(path).read_text()
    for run in range(1, runs + 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if __import__("os").environ.get("THINKING") == "off":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first_response = None
        first_token = None
        events = 0
        text = []
        usage = None
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                first_response = time.perf_counter()
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        continue
                    chunk = json.loads(payload)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        if first_token is None:
                            first_token = time.perf_counter()
                        events += 1
                        text.append(piece)
        except Exception as exc:
            print(f"{path} run{run}: ERROR {exc}", flush=True)
            continue
        end = time.perf_counter()
        n = (usage or {}).get("completion_tokens") or events
        ttft = (first_token or first_response or end) - t0
        wall = end - t0
        decode = max(0.0, wall - ttft)
        tps = (n / decode) if decode > 0 else 0
        print(json.dumps({
            "workload": pathlib.Path(path).stem, "run": run,
            "prompt_chars": len(prompt), "completion_tokens": n,
            "stream_events": events, "ttft_ms": round(ttft * 1000, 2),
            "wall_ms": round(wall * 1000, 2), "decode_tok_s": round(tps, 3),
            "text_chars": len("".join(text)),
        }), flush=True)
