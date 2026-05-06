"""Tests for dashboard index status accounting."""

from __future__ import annotations

import importlib
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


def _write_result(eval_dir: Path, task: str) -> None:
    task_dir = eval_dir / "20260101T000000Z_vA_full" / task / "snapshot"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "results_2026-01-01.json").write_text("x" * 200)


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
    assert variant["evals"]["extra_tasks"] == ["mmlu_generative"]
