"""Tests for results-directory manifest building."""

from __future__ import annotations

import json
from pathlib import Path

from llm_bench.manifest import (
    EvalManifest,
    SpeedManifest,
    eval_is_measured,
    eval_manifest,
    speed_is_measured,
    speed_manifest,
)


def _write_speed_json(d: Path, name: str, **kwargs) -> Path:
    p = d / f"{name}.json"
    p.write_text(json.dumps(kwargs))
    return p


def test_empty_dirs_return_empty(tmp_path):
    sm = speed_manifest(tmp_path / "missing")
    assert isinstance(sm, SpeedManifest)
    assert sm.counts == {} and sm.last_ts == {}
    em = eval_manifest(tmp_path / "missing")
    assert isinstance(em, EvalManifest)
    assert em.measured == set() and em.last_ts == {}


def test_speed_manifest_uses_variant_key_when_present(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_speed_json(raw, "a", variant_key="vA", scenario="p256_g128",
                      bench_version="0.3", run_idx=1, ts="2026-01-01T00:00:00Z")
    _write_speed_json(raw, "b", variant_key="vA", scenario="p256_g128",
                      bench_version="0.3", run_idx=2, ts="2026-01-01T00:01:00Z")
    sm = speed_manifest(raw)
    assert sm.counts[("vA", "p256_g128", "0.3")] == 2
    assert sm.last_ts["vA"] == "2026-01-01T00:01:00Z"


def test_speed_manifest_skips_warmup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_speed_json(raw, "warm", variant_key="vA", scenario="p256_g128",
                      bench_version="0.3", run_idx=0, ts="t0")
    _write_speed_json(raw, "real", variant_key="vA", scenario="p256_g128",
                      bench_version="0.3", run_idx=1, ts="t1")
    sm = speed_manifest(raw)
    assert sm.counts[("vA", "p256_g128", "0.3")] == 1


def test_speed_manifest_rescues_legacy_via_registry(tmp_path):
    """Legacy data without variant_key but with (model_id, fmt, quant)
    should resolve to the registry key for an existing variant."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_speed_json(
        raw, "legacy",
        model_id="gemma-4-26B-A4B-it", fmt="mlx", quant="MLX-8bit",
        scenario="p256_g128", run_idx=1, ts="t1",
    )
    sm = speed_manifest(raw)
    # Should be keyed by registry variant 26B-MoE-mlx-8bit, not fallback
    assert ("26B-MoE-mlx-8bit", "p256_g128", "0.3") in sm.counts


def test_speed_is_measured_default_n3(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(1, 4):
        _write_speed_json(raw, f"r{i}", variant_key="vA", scenario="p256_g128",
                          bench_version="0.3", run_idx=i, ts=f"t{i}")
    sm = speed_manifest(raw)
    assert speed_is_measured(sm.counts, "vA", "p256_g128") is True
    assert speed_is_measured(sm.counts, "vA", "p256_g128", n_required=4) is False


def test_eval_manifest_recognises_runs(tmp_path):
    eval_dir = tmp_path / "eval_scores"
    run_dir = eval_dir / "20260101T000000Z_26B-MoE-mlx-8bit_smoke"
    task_dir = run_dir / "mmlu_generative" / "snapshot"
    task_dir.mkdir(parents=True)
    # Padding to clear the 100-byte minimum that filters trivial empties.
    payload = json.dumps({"results": {"mmlu_generative": {
        "acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "mmlu_generative"}}})
    (task_dir / "results_2026-01-01.json").write_text(payload + " " * 200)
    em = eval_manifest(eval_dir)
    assert ("26B-MoE-mlx-8bit", "mmlu_generative") in em.measured
    assert eval_is_measured(em.measured, "26B-MoE-mlx-8bit", "mmlu_generative")


def test_eval_manifest_normalises_timestamp(tmp_path):
    eval_dir = tmp_path / "eval_scores"
    run_dir = eval_dir / "20260428T080426Z_26B-MoE-mlx-8bit_smoke"
    (run_dir / "task1").mkdir(parents=True)
    (run_dir / "task1" / "results_x.json").write_text("x" * 200)
    em = eval_manifest(eval_dir)
    assert em.last_ts["26B-MoE-mlx-8bit"] == "2026-04-28T08:04:26Z"


def test_eval_manifest_ignores_tiny_results(tmp_path):
    """Results files smaller than 100 bytes are treated as failures."""
    eval_dir = tmp_path / "eval_scores"
    run_dir = eval_dir / "20260101T000000Z_vA_smoke"
    (run_dir / "tiny").mkdir(parents=True)
    (run_dir / "tiny" / "results_x.json").write_text("{}")  # too small
    em = eval_manifest(eval_dir)
    assert em.measured == set()
