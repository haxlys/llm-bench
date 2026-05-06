"""Small reporting helpers shared by dashboards and tests."""

from __future__ import annotations

import pandas as pd


def runtime_column(df: pd.DataFrame) -> str:
    """Prefer generic backend metadata, fall back to legacy fmt."""
    if "backend" in df.columns:
        values = df["backend"].fillna("").astype(str).str.strip()
        if values.ne("").any():
            return "backend"
    return "fmt"


def ordered_variants(df: pd.DataFrame) -> list[str]:
    """Return stable variant order using metadata instead of a hardcoded registry list."""
    if df.empty or "variant" not in df.columns:
        return []
    sort_cols = [
        col for col in ("model_id", "tier", "backend", "fmt", "quant", "variant")
        if col in df.columns
    ]
    variants = df[sort_cols].drop_duplicates()
    return variants.sort_values(sort_cols)["variant"].tolist()
