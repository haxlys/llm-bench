"""Aggregate all lm-eval-harness outputs into results/eval_summary.csv.

Run after a benchmark run completes. Idempotent — safe to re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench.evals.aggregate import load_eval_results, primary_metric_view  # noqa: E402
from llm_bench.index import write_index  # noqa: E402

EVAL_DIR = ROOT / "results" / "eval_scores"
OUT_FULL = ROOT / "results" / "eval_summary_full.csv"
OUT_PRIMARY = ROOT / "results" / "eval_summary_primary.csv"


def main():
    df = load_eval_results(EVAL_DIR)
    if not df.empty:
        df.to_csv(OUT_FULL, index=False)
        primary = primary_metric_view(df)
        primary.to_csv(OUT_PRIMARY, index=False)
        print(f"→ {OUT_FULL.relative_to(ROOT)}: {len(df)} metric rows")
        print(f"→ {OUT_PRIMARY.relative_to(ROOT)}: {len(primary)} (variant × task) rows")
    else:
        print(f"No eval results found under {EVAL_DIR}")
    # Always rebuild the index even if no eval data — speed-only is meaningful.
    idx = write_index()
    print(f"→ {idx.relative_to(ROOT)}: registry × measurement status")


if __name__ == "__main__":
    main()
