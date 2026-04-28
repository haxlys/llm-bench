"""Berkeley Function-Calling Leaderboard runner (skeleton).

BFCL has its own repo and harness:
    https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

For v0.1 we leave BFCL as a manual step — the eval suite reports "not_run" until
the user clones BFCL and points run_evals.py --bfcl-dir at it.
"""

from __future__ import annotations

from pathlib import Path


def run_bfcl(base_url: str, model_label: str, output_dir: Path,
             bfcl_dir: Path | None = None, limit: int | None = None) -> dict:
    if bfcl_dir is None or not bfcl_dir.exists():
        return {
            "task": "bfcl",
            "status": "skipped",
            "reason": "BFCL repo not configured. See src/llm_bench/evals/bfcl.py docstring.",
        }
    # TODO: invoke openfunctions_evaluation.py with --model and --base-url
    return {"task": "bfcl", "status": "not_implemented"}
