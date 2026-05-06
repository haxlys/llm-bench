"""Tests for speed raw result loading."""

from __future__ import annotations

import json

from llm_bench.aggregate import load_raw


def test_load_raw_fills_generic_backend_metadata_for_legacy_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "legacy.json").write_text(json.dumps({
        "ts": "2026-01-01T00:00:00Z",
        "model_id": "m",
        "fmt": "mlx",
        "quant": "MLX-8bit",
        "scenario": "p256_g128",
        "n_prompt": 256,
        "n_gen": 128,
        "pp_tps": 1.0,
        "tg_tps": 2.0,
        "peak_mem_gb": 3.0,
        "wall_s": 4.0,
        "run_idx": 1,
    }))

    df = load_raw(raw_dir)

    assert df.iloc[0]["backend"] == "mlx"
    assert df.iloc[0]["artifact_type"] == "hf_repo"
    assert df.iloc[0]["generation_mode"] == "ar"
