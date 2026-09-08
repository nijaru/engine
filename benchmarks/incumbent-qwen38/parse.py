#!/usr/bin/env python3
"""Parse llama-server logs in results/*/server.log into a summary table.

Extracts per-completion timing blocks and speculative decoding stats.
Usage: parse.py results/*/server.log
"""
import re
import sys
from pathlib import Path

TIMING = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
    re.S)
# spec stats lines look like: "spec: ... accepted = N/M (...)" or draft/accept summaries
SPEC = re.compile(r"spec\w*\s*:.*", re.I)


def parse(path: Path):
    text = path.read_text(errors="replace")
    runs = []
    for m in TIMING.finditer(text):
        runs.append({
            "prompt_tps": float(m.group(2)) / (float(m.group(1)) / 1000.0),
            "prompt_tokens": int(m.group(2)),
            "eval_tokens": int(m.group(4)),
            "eval_tps": float(m.group(5)),
        })
    spec_lines = [l.strip() for l in text.splitlines() if SPEC.match(l)]
    return runs, spec_lines


def main():
    for p in map(Path, sys.argv[1:]):
        tag = p.parent.name
        runs, spec = parse(p)
        if not runs:
            print(f"{tag}: no timing blocks found")
            continue
        print(f"== {tag} ({len(runs)} completions)")
        for i, r in enumerate(runs):
            print(f"  run{i+1}: prompt={r['prompt_tps']:.0f} t/s ({r['prompt_tokens']} tok) "
                  f"eval={r['eval_tps']:.1f} t/s ({r['eval_tokens']} tok)")
        for s in spec[-6:]:
            print(f"  {s[:160]}")


if __name__ == "__main__":
    main()
