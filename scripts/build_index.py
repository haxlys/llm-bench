"""Generate results/index.json — registry × measurement status snapshot."""

from __future__ import annotations

from pathlib import Path

from llm_bench.index import write_index

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    out = write_index()
    print(f"→ {out.relative_to(ROOT)}")
