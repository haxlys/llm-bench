"""Aggregate raw measurement JSONs into a single summary CSV.

Tier comes from the registry — no hardcoded MLX-8bit/Q8_0 → 8bit map here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_bench.registry import default_artifact_type, get_registry


SUMMARY_COLS = [
    "ts", "model_id", "fmt", "backend", "artifact_type", "quant", "scenario",
    "n_prompt", "n_gen", "pp_tps", "tg_tps",
    "peak_mem_gb", "wall_s", "run_idx", "bench_version", "variant_key",
]


def _quant_to_tier(quant: str) -> str:
    """Look up tier from registry. Returns '' if not found."""
    for v in get_registry().variants:
        if v.quant == quant:
            return v.tier
    return ""


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        row = {k: data.get(k) for k in SUMMARY_COLS}
        if not row.get("backend"):
            row["backend"] = row.get("fmt", "")
        if not row.get("artifact_type"):
            row["artifact_type"] = default_artifact_type(row.get("fmt", ""))
        row["tier"] = _quant_to_tier(row.get("quant", ""))
        rows.append(row)
    cols = SUMMARY_COLS + ["tier"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def write_summary(raw_dir: Path, out_csv: Path) -> Path:
    df = load_raw(raw_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def aggregate_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average across run_idx (excluding warmup, run_idx >= 1)."""
    if df.empty:
        return df
    real = df[df["run_idx"] >= 1].copy()
    group_cols = [
        "model_id", "fmt", "backend", "artifact_type", "quant",
        "scenario", "n_prompt", "n_gen",
    ]
    grouped = real.groupby(group_cols, as_index=False).agg(
        pp_tps_mean=("pp_tps", "mean"),
        pp_tps_std=("pp_tps", "std"),
        tg_tps_mean=("tg_tps", "mean"),
        tg_tps_std=("tg_tps", "std"),
        peak_mem_gb_mean=("peak_mem_gb", "mean"),
        wall_s_mean=("wall_s", "mean"),
        n_runs=("run_idx", "count"),
    )
    grouped["tier"] = grouped["quant"].map(_quant_to_tier)
    return grouped
