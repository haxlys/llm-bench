"""Tests for dashboard/reporting helpers."""

from __future__ import annotations

import pandas as pd

from llm_bench.reporting import ordered_variants, runtime_column


def test_runtime_column_prefers_backend_when_available():
    df = pd.DataFrame({"fmt": ["api"], "backend": ["openai-compatible"]})

    assert runtime_column(df) == "backend"


def test_runtime_column_falls_back_to_fmt_for_legacy_frames():
    df = pd.DataFrame({"fmt": ["mlx"], "backend": [""]})

    assert runtime_column(df) == "fmt"


def test_ordered_variants_uses_metadata_instead_of_hardcoded_keys():
    df = pd.DataFrame(
        [
            {
                "variant": "z-hosted",
                "model_id": "m2",
                "tier": "hosted",
                "backend": "openai-compatible",
                "quant": "hosted",
            },
            {
                "variant": "a-local",
                "model_id": "m1",
                "tier": "8bit",
                "backend": "gguf",
                "quant": "Q8_0",
            },
        ]
    )

    assert ordered_variants(df) == ["a-local", "z-hosted"]
