"""Tests for MTPLX on/off speedup reporting."""

from __future__ import annotations

import csv
import json

from llm_bench.mtplx_compare import write_speedup_report


def _raw(
    raw_dir,
    name: str,
    variant_key: str,
    generation_mode: str,
    tg_tps: float,
    accept_rates: list[float] | None = None,
):
    payload = {
        "model_id": "qwen-3.6-27B-MTPLX",
        "fmt": "mlx",
        "backend": "mtplx",
        "artifact_type": "hf_repo",
        "quant": "MLX-4bit",
        "scenario": "p256_g128",
        "generation_mode": generation_mode,
        "n_prompt": 256,
        "n_gen": 128,
        "pp_tps": 0.0,
        "tg_tps": tg_tps,
        "peak_mem_gb": 15.0,
        "wall_s": 7.0,
        "run_idx": 1,
        "bench_version": "0.3",
        "variant_key": variant_key,
        "raw": {
            "acceptance_rates_by_depth": accept_rates or [],
            "verify_ms_per_call": 50.0 if generation_mode == "mtp" else None,
        },
    }
    (raw_dir / name).write_text(json.dumps(payload))


def test_write_speedup_report_pairs_mtp_and_ar_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw(raw_dir, "mtp1.json", "model-speed-mtplx-mtp", "mtp", 42.0, [0.8, 0.6])
    _raw(raw_dir, "mtp2.json", "model-speed-mtplx-mtp", "mtp", 44.0, [0.9, 0.5])
    _raw(raw_dir, "ar1.json", "model-speed-mtplx-ar", "ar", 22.0)

    out = tmp_path / "mtplx_speedups.csv"
    write_speedup_report(raw_dir, out)

    rows = list(csv.DictReader(out.open()))

    assert len(rows) == 1
    row = rows[0]
    assert row["pair_key"] == "model-speed-mtplx"
    assert row["scenario"] == "p256_g128"
    assert row["mtp_runs"] == "2"
    assert row["ar_runs"] == "1"
    assert row["mtp_tg_tps_mean"] == "43.000"
    assert row["ar_tg_tps_mean"] == "22.000"
    assert row["speedup"] == "1.955"
    assert row["accept_d1_mean"] == "0.850"
    assert row["accept_d2_mean"] == "0.550"
    assert row["verify_ms_per_call_mean"] == "50.000"
