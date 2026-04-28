"""Idempotency for benchmark runs.

Walks results/raw/*.json and counts how many measured runs (run_idx >= 1) exist
for each (variant_key, scenario, bench_version). Used by run_bench.py to skip
combos that already have N successful runs.

For eval results, we mirror the same idea against results/eval_scores/<run_id>/
directories — a (variant_key, task, bench_version) triple is "measured" when
a non-empty results_*.json exists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from llm_bench import BENCH_VERSION
from llm_bench.registry import get_registry


# ---------- speed (run_bench.py) ----------

def _variant_key_from_meta(model_id: str, fmt: str, quant: str) -> str | None:
    """Look up registry variant key by metadata triple. Returns None if not found."""
    for v in get_registry().variants:
        if v.model_id == model_id and v.fmt == fmt and v.quant == quant:
            return v.key
    return None


def speed_manifest(raw_dir: Path) -> dict[tuple, int]:
    """Returns {(variant_key, scenario, bench_version): measured_count}.

    Measured = run_idx >= 1 (warmup excluded). Files missing variant_key are
    rescued by registry lookup on (model_id, fmt, quant).
    Files missing bench_version are treated as the current BENCH_VERSION (the
    historical data was collected with the same methodology).
    """
    counts: dict[tuple, int] = defaultdict(int)
    if not raw_dir.exists():
        return counts
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("run_idx", 0) < 1:
            continue
        vkey = data.get("variant_key") or _variant_key_from_meta(
            data.get("model_id", ""), data.get("fmt", ""), data.get("quant", "")
        ) or _fallback_key(data)
        bv = data.get("bench_version") or BENCH_VERSION  # legacy data → current
        counts[(vkey, data.get("scenario", ""), bv)] += 1
    return counts


def _fallback_key(data: dict) -> str:
    """Last-resort key when registry lookup fails."""
    return f"{data.get('model_id','?')}__{data.get('fmt','?')}__{data.get('quant','?')}"


def speed_is_measured(
    counts: dict[tuple, int],
    variant_key: str,
    scenario: str,
    n_required: int = 3,
    bench_version: str = BENCH_VERSION,
) -> bool:
    return counts.get((variant_key, scenario, bench_version), 0) >= n_required


# ---------- evals (run_evals.py) ----------

_RUN_DIR_RE = __import__("re").compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_(?P<variant>[\w-]+?)_(?P<suite>smoke|full)$"
)


def eval_manifest(eval_dir: Path) -> set[tuple]:
    """Returns set of (variant_key, task) pairs that have at least one
    non-empty results_*.json file. Run-dir names follow the
    `<ts>_<variant>_<suite>` pattern.
    """
    measured: set[tuple] = set()
    if not eval_dir.exists():
        return measured
    for run_dir in eval_dir.iterdir():
        if not run_dir.is_dir():
            continue
        m = _RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        variant_key = m.group("variant")
        for task_dir in run_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if any(p.stat().st_size > 100 for p in task_dir.rglob("results_*.json")):
                measured.add((variant_key, task))
    return measured


def eval_is_measured(
    measured: set[tuple],
    variant_key: str,
    task: str,
) -> bool:
    return (variant_key, task) in measured
