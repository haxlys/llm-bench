"""Append-only JSONL traces for eval task runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_trace(path: Path, record: dict) -> Path:
    """Append one eval trace record as JSONL.

    Trace rows are intentionally runner-agnostic so they can act as a compact
    ledger across lm-eval, EvalPlus, LiveCodeBench, BFCL, and source QA.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
