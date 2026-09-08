#!/usr/bin/env python3
"""Generate filled-context workload variants for context-length scaling.

Builds code-task prompts whose context is padded to ~N tokens with realistic,
repetitive code (imports, helpers, repeated identifiers) so speculative
acceptance reflects coding-agent sessions, not random filler.
"""
import sys
from pathlib import Path

FILLER = '''
# --- module: pipeline_stage_{i} ---
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("pipeline.stage_{i}")

MAX_RETRIES = 3
DEFAULT_TIMEOUT_MS = 2500
STAGE_NAMES = ["ingest", "validate", "transform", "enrich", "emit"]


@dataclass
class StageConfig:
    name: str = "stage_{i}"
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    max_retries: int = MAX_RETRIES
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "tags": list(self.tags),
            "enabled": self.enabled,
        }


class StageError(Exception):
    def __init__(self, stage: str, message: str, retryable: bool = False):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.retryable = retryable


def run_stage_{i}(payload: dict, config: Optional[StageConfig] = None) -> dict:
    cfg = config or StageConfig()
    records = payload.get("records", [])
    out = []
    for rec in records:
        if not cfg.enabled:
            break
        try:
            transformed = {k: v for k, v in rec.items() if k not in ("_raw", "_meta")}
            transformed["stage"] = cfg.name
            out.append(transformed)
        except KeyError as exc:
            raise StageError(cfg.name, f"missing key: {exc}", retryable=False) from exc
    logger.debug("stage_%s processed %d records", "{i}", len(out))
    return {"records": out, "stage": cfg.name, "ok": True}
'''

TASK = """

Based on the pipeline modules above, write a new module `pipeline_aggregator` that:
1. Collects results from run_stage_0 through run_stage_{last} with the retry semantics shown,
2. Aggregates per-stage record counts into a summary dict,
3. Stops at the first non-retryable StageError and returns partial results plus the error.
Emit the complete module as a single code block.
"""


def main(target_tokens: int, out_dir: str, tok_per_rep: int = 260):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_reps = max(1, target_tokens // tok_per_rep)
    body = "".join(FILLER.replace("{i}", str(i)) for i in range(n_reps))
    text = body + TASK.replace("{last}", str(n_reps - 1))
    p = out / f"code-fill{target_tokens}.txt"
    p.write_text(text)
    print(f"{p} ~{target_tokens} tokens target, {len(text)} chars")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(int(t), sys.argv[2] if len(sys.argv) > 2 else "workloads")
