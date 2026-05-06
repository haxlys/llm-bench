from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_bench.site_data import build_site_data, parse_scenario, write_site_data


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
defaults: {}
models:
  - id: test-model
    family: qwen
    architecture: dense
    params_total_b: 4
    params_active_b: 4
    variants:
      - key: test-variant
        fmt: gguf
        path: /tmp/test.gguf
        quant: Q8_0
        tier: 8bit
        approx_size_gb: 4.5
  - id: mtplx-model
    family: qwen
    architecture: dense
    params_total_b: 27
    params_active_b: 27
    variants:
      - key: mtplx-mtp
        fmt: mlx
        backend: mtplx
        generation_mode: mtp
        path: org/mtp
        quant: MLX-4bit
        tier: 4bit
      - key: mtplx-ar
        fmt: mlx
        backend: mtplx
        generation_mode: ar
        path: org/ar
        quant: MLX-4bit
        tier: 4bit
""".strip()
    )


def test_parse_scenario_extracts_prompt_and_generation_tokens() -> None:
    assert parse_scenario("p4096_g512") == (4096, 512)


def test_build_site_data_preserves_zero_and_marks_unmeasured_latency(tmp_path: Path) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_csv(
        tmp_path / "results" / "summary.csv",
        [
            {
                "ts": "2026-05-06T00:00:00+00:00",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "scenario": "p256_g128",
                "generation_mode": "ar",
                "n_prompt": 256,
                "n_gen": 128,
                "pp_tps": 100.0,
                "tg_tps": 50.0,
                "peak_mem_gb": 4.5,
                "wall_s": 2.0,
                "run_idx": 1,
                "bench_version": "0.3",
                "variant_key": "test-variant",
                "tier": "8bit",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "eval_summary_primary.csv",
        [
            {
                "variant": "test-variant",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "tier": "8bit",
                "family": "qwen",
                "architecture": "dense",
                "dim": "reasoning",
                "task": "gsm8k_cot_zeroshot",
                "run_id": "run-1",
                "ts": "20260506T000000Z",
                "subtask": "gsm8k_cot_zeroshot",
                "metric": "exact_match,flexible-extract",
                "value": 0.0,
                "stderr": 0.0,
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "mtplx_speedups.csv",
        [
            {
                "pair_key": "mtplx-pair",
                "scenario": "p256_g128",
                "mtp_variant": "mtplx-mtp",
                "ar_variant": "mtplx-ar",
                "mtp_runs": 3,
                "ar_runs": 3,
                "mtp_tg_tps_mean": 40.0,
                "ar_tg_tps_mean": 25.0,
                "speedup": 1.6,
                "mtp_peak_mem_gb_mean": 15.0,
                "ar_peak_mem_gb_mean": 15.1,
                "verify_ms_per_call_mean": 51.0,
                "accept_d1_mean": 0.8,
                "accept_d2_mean": 0.6,
                "accept_d3_mean": 0.4,
            }
        ],
    )

    data = build_site_data(
        repo_root=tmp_path,
        generated_at="2026-05-06T12:00:00+00:00",
        source_commit="abc1234",
    )

    assert data["benchVersion"] == "0.3"
    assert data["sourceCommit"] == "abc1234"
    assert data["accuracy"][0]["value"] == 0.0
    assert data["accuracy"][0]["confidence"] == "directional"
    assert data["speed"][0]["ppTpsMean"] == 100.0
    assert data["speed"][0]["tgTpsMean"] == 50.0
    assert data["speed"][0]["ttftMs"] is None
    assert data["speed"][0]["itlMs"] is None
    assert data["mtplx"][0]["speedup"] == 1.6


def test_write_site_data_outputs_stable_json(tmp_path: Path) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_csv(
        tmp_path / "results" / "summary.csv",
        [
            {
                "ts": "2026-05-06T00:00:00+00:00",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "scenario": "p256_g128",
                "generation_mode": "ar",
                "n_prompt": 256,
                "n_gen": 128,
                "pp_tps": 100.0,
                "tg_tps": 50.0,
                "peak_mem_gb": 4.5,
                "wall_s": 2.0,
                "run_idx": 1,
                "bench_version": "0.3",
                "variant_key": "test-variant",
                "tier": "8bit",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "eval_summary_primary.csv",
        [
            {
                "variant": "test-variant",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "tier": "8bit",
                "family": "qwen",
                "architecture": "dense",
                "dim": "source_grounding",
                "task": "sourceqa",
                "run_id": "run-1",
                "ts": "20260506T000000Z",
                "subtask": "sourceqa",
                "metric": "acc,none",
                "value": 1.0,
                "stderr": "",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "mtplx_speedups.csv",
        [
            {
                "pair_key": "mtplx-pair",
                "scenario": "p256_g128",
                "mtp_variant": "mtplx-mtp",
                "ar_variant": "mtplx-ar",
                "mtp_runs": 1,
                "ar_runs": 1,
                "mtp_tg_tps_mean": 40.0,
                "ar_tg_tps_mean": 25.0,
                "speedup": 1.6,
                "mtp_peak_mem_gb_mean": 15.0,
                "ar_peak_mem_gb_mean": 15.1,
                "verify_ms_per_call_mean": 51.0,
                "accept_d1_mean": 0.8,
                "accept_d2_mean": 0.6,
                "accept_d3_mean": 0.4,
            }
        ],
    )

    out = tmp_path / "site" / "src" / "data" / "benchmarks.json"
    write_site_data(tmp_path, out, generated_at="2026-05-06T12:00:00+00:00")

    parsed = json.loads(out.read_text())
    assert parsed["generatedAt"] == "2026-05-06T12:00:00+00:00"
    assert parsed["variants"][0]["key"] == "test-variant"
