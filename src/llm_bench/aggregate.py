"""Aggregate raw measurement JSONs into a single summary CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SUMMARY_COLS = [
    "ts", "model_id", "fmt", "quant", "scenario",
    "n_prompt", "n_gen", "pp_tps", "tg_tps",
    "peak_mem_gb", "wall_s", "run_idx",
]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({k: data.get(k) for k in SUMMARY_COLS})
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLS)
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
    grouped = real.groupby(
        ["model_id", "fmt", "quant", "scenario", "n_prompt", "n_gen"], as_index=False
    ).agg(
        pp_tps_mean=("pp_tps", "mean"),
        pp_tps_std=("pp_tps", "std"),
        tg_tps_mean=("tg_tps", "mean"),
        tg_tps_std=("tg_tps", "std"),
        peak_mem_gb_mean=("peak_mem_gb", "mean"),
        wall_s_mean=("wall_s", "mean"),
        n_runs=("run_idx", "count"),
    )
    return grouped
