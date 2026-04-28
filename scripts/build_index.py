"""Generate results/index.json — registry × measurement status snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench.index import write_index  # noqa: E402

if __name__ == "__main__":
    out = write_index()
    print(f"→ {out.relative_to(ROOT)}")
