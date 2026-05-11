from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from llm_bench.site_data import SiteDataError, build_site_data, parse_scenario, write_site_data


def _load_export_site_public_data_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_site_public_data.py"
    spec = importlib.util.spec_from_file_location("export_site_public_data", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _write_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "key": "test-variant",
                        "evals": {
                            "coverage": [
                                {
                                    "dim": "source_grounding",
                                    "task": "sourceqa",
                                    "runner": "sourceqa",
                                    "lane": "primary",
                                    "required": True,
                                    "supported": True,
                                    "measured": True,
                                    "confidence": "measured",
                                    "status": "measured",
                                },
                                {
                                    "dim": "agentic_code",
                                    "task": "programbench",
                                    "runner": "programbench",
                                    "lane": "optional",
                                    "required": False,
                                    "supported": True,
                                    "measured": False,
                                    "confidence": "measured",
                                    "status": "optional",
                                },
                            ]
                        },
                    }
                ]
            }
        )
    )


def _speed_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    row.update(overrides)
    return row


def _accuracy_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    row.update(overrides)
    return row


def _mtplx_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    row.update(overrides)
    return row


def _write_site_inputs(
    tmp_path: Path,
    *,
    speed_rows: list[dict[str, object]] | None = None,
    accuracy_rows: list[dict[str, object]] | None = None,
    mtplx_rows: list[dict[str, object]] | None = None,
) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_index(tmp_path / "results" / "index.json")
    _write_csv(tmp_path / "results" / "summary.csv", speed_rows or [_speed_row()])
    _write_csv(
        tmp_path / "results" / "eval_summary_primary.csv",
        accuracy_rows or [_accuracy_row()],
    )
    _write_csv(tmp_path / "results" / "mtplx_speedups.csv", mtplx_rows or [_mtplx_row()])


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
            },
            {
                "ts": "2026-05-06T00:05:00+00:00",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "scenario": "p256_g128",
                "generation_mode": "ar",
                "n_prompt": 256,
                "n_gen": 128,
                "pp_tps": 120.0,
                "tg_tps": 70.0,
                "peak_mem_gb": 5.5,
                "wall_s": 4.0,
                "run_idx": 2,
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
    assert data["speed"][0]["ppTpsMean"] == 110.0
    assert data["speed"][0]["tgTpsMean"] == 60.0
    assert data["speed"][0]["peakMemGbMean"] == 5.0
    assert data["speed"][0]["wallSecondsMean"] == 3.0
    assert data["speed"][0]["runs"] == 2
    assert data["speed"][0]["runIndices"] == [1, 2]
    assert data["speed"][0]["ttftMs"] is None
    assert data["speed"][0]["itlMs"] is None
    assert data["mtplx"][0]["speedup"] == 1.6


def test_build_site_data_uses_bench_version_from_most_recent_speed_row(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_csv(
        tmp_path / "results" / "summary.csv",
        [
            {
                "ts": "2026-05-05T00:00:00+00:00",
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
                "bench_version": "0.9",
                "variant_key": "test-variant",
                "tier": "8bit",
            },
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
                "run_idx": 2,
                "bench_version": "0.10",
                "variant_key": "test-variant",
                "tier": "8bit",
            },
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

    data = build_site_data(
        repo_root=tmp_path,
        generated_at="2026-05-06T12:00:00+00:00",
        source_commit="abc1234",
    )

    assert data["benchVersion"] == "0.10"
    assert data["speed"][0]["benchVersion"] == "0.10"


def test_build_site_data_wraps_missing_registry_as_site_data_error(tmp_path: Path) -> None:
    with pytest.raises(SiteDataError):
        build_site_data(repo_root=tmp_path)


def test_build_site_data_rejects_accuracy_model_id_mismatch(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path, accuracy_rows=[_accuracy_row(model_id="wrong-model")])

    with pytest.raises(SiteDataError, match="model_id"):
        build_site_data(repo_root=tmp_path)


def test_build_site_data_rejects_unknown_accuracy_variant(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path, accuracy_rows=[_accuracy_row(variant="missing-variant")])

    with pytest.raises(SiteDataError, match="unknown variant"):
        build_site_data(repo_root=tmp_path)


def test_programbench_accuracy_rows_carry_agentic_caveat(tmp_path: Path) -> None:
    _write_site_inputs(
        tmp_path,
        accuracy_rows=[
            _accuracy_row(
                dim="agentic_code",
                task="programbench",
                subtask="programbench",
                metric="resolved_rate,none",
                value=0.2,
            )
        ],
    )

    data = build_site_data(repo_root=tmp_path)

    assert data["accuracy"][0]["confidence"] == "measured"
    assert data["accuracy"][0]["caveats"] == ["agentic-scaffold-dependent"]
    assert {"id": "agentic-scaffold-dependent", "status": "measured"} in data["caveats"]
    assert {"id": "mtplx-speed-only", "status": "speed_only"} in data["caveats"]


def test_build_site_data_exports_index_coverage(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path)

    data = build_site_data(repo_root=tmp_path)

    assert data["coverage"][0]["task"] == "sourceqa"
    assert data["coverage"][0]["status"] == "measured"
    assert data["coverage"][1]["task"] == "programbench"
    assert data["coverage"][1]["status"] == "optional"


def test_build_site_data_rejects_invalid_speed_scenario(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path, speed_rows=[_speed_row(scenario="bad_scenario")])

    with pytest.raises(SiteDataError, match="invalid scenario"):
        build_site_data(repo_root=tmp_path)


def test_build_site_data_rejects_invalid_numeric_speed_value(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path, speed_rows=[_speed_row(pp_tps="not-a-number")])

    with pytest.raises(SiteDataError, match="invalid numeric value"):
        build_site_data(repo_root=tmp_path)


def test_build_site_data_rejects_non_finite_speed_value(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path, speed_rows=[_speed_row(tg_tps="NaN")])

    with pytest.raises(SiteDataError, match="non-finite numeric value"):
        build_site_data(repo_root=tmp_path)


def test_build_site_data_wraps_missing_required_csv_as_site_data_error(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")

    with pytest.raises(SiteDataError, match="summary.csv"):
        build_site_data(repo_root=tmp_path)


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


def test_sync_site_public_data_copies_json_and_downloads(tmp_path: Path) -> None:
    _write_site_inputs(tmp_path)
    module = _load_export_site_public_data_module()

    copied = module.sync_site_public_data(tmp_path, generated_at="2026-05-06T12:00:00+00:00")

    src_json = tmp_path / "site" / "src" / "data" / "benchmarks.json"
    public_json = tmp_path / "site" / "public" / "data" / "benchmarks.json"
    public_dir = tmp_path / "site" / "public" / "data"
    assert src_json.read_bytes() == public_json.read_bytes()
    assert (public_dir / "summary.csv").read_bytes() == (
        tmp_path / "results" / "summary.csv"
    ).read_bytes()
    assert (public_dir / "eval_summary_primary.csv").read_bytes() == (
        tmp_path / "results" / "eval_summary_primary.csv"
    ).read_bytes()
    assert (public_dir / "mtplx_speedups.csv").read_bytes() == (
        tmp_path / "results" / "mtplx_speedups.csv"
    ).read_bytes()
    assert (public_dir / "index.json").read_bytes() == (
        tmp_path / "results" / "index.json"
    ).read_bytes()
    assert copied == [
        src_json,
        public_json,
        public_dir / "summary.csv",
        public_dir / "eval_summary_primary.csv",
        public_dir / "mtplx_speedups.csv",
        public_dir / "index.json",
    ]
