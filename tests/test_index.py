"""Tests for dashboard index status accounting."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


class _FakeVariant:
    key = "vA"
    model_id = "m1"
    fmt = "mlx"
    quant = "MLX-8bit"
    tier = "8bit"
    family = "f"
    architecture = "dense"
    approx_size_gb = 1

    def exists_locally(self) -> bool:
        return True


class _FakeRegistry:
    variants = [_FakeVariant()]


class _FakeSpeedOnlyVariant(_FakeVariant):
    key = "vSpeed"
    backend = "mtplx"
    generation_mode = "mtp"


class _FakeSpeedOnlyRegistry:
    variants = [_FakeSpeedOnlyVariant()]


def _write_result(eval_dir: Path, task: str) -> None:
    task_dir = eval_dir / "20260101T000000Z_vA_full" / task / "snapshot"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "results_2026-01-01.json").write_text(
        json.dumps({"results": {task: {"acc,none": 1.0}}})
    )


def test_eval_progress_counts_supported_tasks_only(tmp_path, monkeypatch):
    index_mod = importlib.import_module("llm_bench.index")
    monkeypatch.setattr(index_mod, "ROOT", tmp_path)
    monkeypatch.setattr(index_mod, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(index_mod, "full_suite", lambda: [("reasoning", "gsm8k_cot_zeroshot")])
    monkeypatch.setattr(index_mod, "external_suite", lambda: [])

    eval_dir = tmp_path / "results" / "eval_scores"
    _write_result(eval_dir, "gsm8k_cot_zeroshot")
    _write_result(eval_dir, "mmlu_generative")

    variant = index_mod.build_index()["variants"][0]

    assert variant["evals"]["tasks_supported"] == 1
    assert variant["evals"]["tasks_measured"] == 1
    assert variant["evals"]["tasks"] == ["gsm8k_cot_zeroshot"]
    assert variant["evals"]["coverage"][0]["task"] == "gsm8k_cot_zeroshot"
    assert variant["evals"]["coverage"][0]["status"] == "directional"
    assert variant["evals"]["coverage_summary"]["optional"] == 1
    assert variant["evals"]["extra_tasks"] == ["mmlu_generative"]


def test_mtplx_speed_only_variants_do_not_create_eval_debt(tmp_path, monkeypatch):
    index_mod = importlib.import_module("llm_bench.index")
    monkeypatch.setattr(index_mod, "ROOT", tmp_path)
    monkeypatch.setattr(index_mod, "get_registry", lambda: _FakeSpeedOnlyRegistry())
    monkeypatch.setattr(index_mod, "full_suite", lambda: [("reasoning", "gsm8k_cot_zeroshot")])
    monkeypatch.setattr(index_mod, "external_suite", lambda: [])

    variant = index_mod.build_index()["variants"][0]

    assert variant["evals"]["tasks_supported"] == 0
    assert variant["evals"]["coverage"][0]["lane"] == "mtplx_speedup"
    assert variant["evals"]["coverage"][0]["required"] is False
    assert variant["evals"]["coverage"][0]["status"] == "speed_only"
    assert variant["evals"]["coverage_summary"]["missing"] == 0
    assert variant["evals"]["coverage_summary"]["speed_only"] == 2


def test_diagnostic_sourceqa_does_not_create_primary_eval_debt(tmp_path, monkeypatch):
    index_mod = importlib.import_module("llm_bench.index")
    monkeypatch.setattr(index_mod, "ROOT", tmp_path)
    monkeypatch.setattr(index_mod, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(index_mod, "full_suite", lambda: [])
    monkeypatch.setattr(index_mod, "external_suite", lambda: [("diagnostic", "sourceqa", "sourceqa")])

    variant = index_mod.build_index()["variants"][0]

    assert variant["evals"]["tasks_supported"] == 0
    assert variant["evals"]["coverage"][0]["lane"] == "diagnostic"
    assert variant["evals"]["coverage"][0]["required"] is False
    assert variant["evals"]["coverage"][0]["status"] == "diagnostic"
    assert variant["evals"]["coverage_summary"]["diagnostic"] == 1
    assert variant["evals"]["coverage_summary"]["missing"] == 0
