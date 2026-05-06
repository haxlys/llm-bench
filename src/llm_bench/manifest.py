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
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from llm_bench import BENCH_VERSION, N_RUNS_REQUIRED
from llm_bench.registry import get_registry


# ---------- speed (run_bench.py) ----------

@lru_cache(maxsize=1)
def _meta_to_key() -> dict[tuple, str]:
    """Cached (model_id, fmt, quant) → variant_key map. O(1) lookup."""
    return {(v.model_id, v.fmt, v.quant): v.key for v in get_registry().variants}


@dataclass
class SpeedManifest:
    """Compact summary of results/raw/ — single read, two views.

    Attributes:
        counts: {(variant_key, scenario, bench_version): measured_count}
                Measured = run_idx >= 1 (warmup excluded).
        last_ts: {variant_key: latest ts seen for that variant}
                 ISO 8601 timestamps; "" for variants with no data.
    """
    counts: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))
    last_ts: dict[str, str] = field(default_factory=dict)


def _fallback_key(data: dict) -> str:
    """Last-resort key when registry lookup fails."""
    return f"{data.get('model_id','?')}__{data.get('fmt','?')}__{data.get('quant','?')}"


def _resolve_variant_key(data: dict) -> str:
    return (
        data.get("variant_key")
        or _meta_to_key().get(
            (data.get("model_id", ""), data.get("fmt", ""), data.get("quant", ""))
        )
        or _fallback_key(data)
    )


def speed_manifest(raw_dir: Path) -> SpeedManifest:
    """Single pass over raw_dir/*.json producing both count and last-ts views.

    Files missing variant_key are rescued via registry lookup on
    (model_id, fmt, quant). Files missing bench_version are treated as the
    current BENCH_VERSION (legacy data → assumed compatible).
    """
    out = SpeedManifest()
    if not raw_dir.exists():
        return out
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        vkey = _resolve_variant_key(data)
        ts = data.get("ts", "")
        if ts and ts > out.last_ts.get(vkey, ""):
            out.last_ts[vkey] = ts
        if data.get("run_idx", 0) < 1:
            continue
        bv = data.get("bench_version") or BENCH_VERSION
        out.counts[(vkey, data.get("scenario", ""), bv)] += 1
    return out


def speed_is_measured(
    counts: dict[tuple, int],
    variant_key: str,
    scenario: str,
    n_required: int = N_RUNS_REQUIRED,
    bench_version: str = BENCH_VERSION,
) -> bool:
    return counts.get((variant_key, scenario, bench_version), 0) >= n_required


# ---------- evals (run_evals.py) ----------

# Run-id directory name: <ts>_<variant>_<suite>. Single source of truth —
# evals.aggregate re-imports this so a format change updates both readers.
RUN_DIR_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_(?P<variant>[\w.-]+?)_(?P<suite>smoke|full)$"
)
_RUN_DIR_RE = RUN_DIR_RE  # legacy alias — keep until external code migrates


@dataclass
class EvalManifest:
    """Single-pass view of results/eval_scores/.

    Attributes:
        measured: {(variant_key, task)} pairs with non-empty results JSON.
        last_ts:  {variant_key: latest run-dir ts seen} as ISO 8601.
    """
    measured: set[tuple] = field(default_factory=set)
    last_ts: dict[str, str] = field(default_factory=dict)


def _iso_from_dir_ts(ts: str) -> str:
    """'20260428T080426Z' → '2026-04-28T08:04:26Z'."""
    if len(ts) >= 16 and ts[8] == "T":
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"
    return ts


def eval_manifest(eval_dir: Path) -> EvalManifest:
    """Walk eval_scores/, returning measured (variant, task) pairs + last ts."""
    out = EvalManifest()
    if not eval_dir.exists():
        return out
    for run_dir in eval_dir.iterdir():
        if not run_dir.is_dir():
            continue
        m = _RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        variant_key = m.group("variant")
        ts_iso = _iso_from_dir_ts(m.group("ts"))
        if ts_iso > out.last_ts.get(variant_key, ""):
            out.last_ts[variant_key] = ts_iso
        for task_dir in run_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if any(p.stat().st_size > 100 for p in task_dir.rglob("results_*.json")):
                out.measured.add((variant_key, task))
    return out


def eval_is_measured(
    measured: set[tuple],
    variant_key: str,
    task: str,
) -> bool:
    return (variant_key, task) in measured
