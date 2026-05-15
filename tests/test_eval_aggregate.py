"""Tests for eval result aggregation metadata."""

from __future__ import annotations

import json

import llm_bench.evals.aggregate as aggregate_mod
from llm_bench.evals.aggregate import load_eval_results, primary_metric_view


class _FakeVariant:
    model_id = "provider/model"
    fmt = "api"
    backend = "openai-compatible"
    artifact_type = "endpoint"
    quant = "hosted"
    tier = "hosted"
    family = "hosted"
    architecture = "dense"


class _FakeRegistry:
    def variant(self, key: str):
        assert key == "vA"
        return _FakeVariant()


def test_eval_results_keep_backend_and_artifact_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    task_dir = tmp_path / "20260101T000000Z_vA_full" / "sourceqa" / "snapshot"
    task_dir.mkdir(parents=True)
    (task_dir / "results_2026-01-01_sourceqa.json").write_text(
        json.dumps({"results": {"sourceqa": {"acc,none": 0.75}}})
    )

    full = load_eval_results(tmp_path)
    primary = primary_metric_view(full)

    assert full.iloc[0]["backend"] == "openai-compatible"
    assert full.iloc[0]["artifact_type"] == "endpoint"
    assert primary.iloc[0]["backend"] == "openai-compatible"
    assert primary.iloc[0]["artifact_type"] == "endpoint"


def test_primary_metric_view_prefers_latest_run(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    old_task = tmp_path / "20260101T000000Z_vA_full" / "sourceqa" / "snapshot"
    new_task = tmp_path / "20260102T000000Z_vA_full" / "sourceqa" / "snapshot"
    old_task.mkdir(parents=True)
    new_task.mkdir(parents=True)
    (old_task / "results_old.json").write_text(
        json.dumps({"results": {"sourceqa": {"acc,none": 0.25}}})
    )
    (new_task / "results_new.json").write_text(
        json.dumps({"results": {"sourceqa": {"acc,none": 0.75}}})
    )

    primary = primary_metric_view(load_eval_results(tmp_path))

    assert primary.iloc[0]["value"] == 0.75
    assert primary.iloc[0]["run_id"] == "20260102T000000Z_vA_full"


def test_primary_metric_view_latest_subtask_aggregate_keeps_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    old_task = tmp_path / "20260101T000000Z_vA_full" / "hrm8k" / "snapshot"
    new_task = tmp_path / "20260102T000000Z_vA_full" / "hrm8k" / "snapshot"
    old_task.mkdir(parents=True)
    new_task.mkdir(parents=True)
    (old_task / "results_old.json").write_text(
        json.dumps({"results": {"hrm8k_a": {"exact_match,none": 0.0}}})
    )
    (new_task / "results_new.json").write_text(
        json.dumps({
            "results": {
                "hrm8k_a": {"exact_match,none": 0.5},
                "hrm8k_b": {"exact_match,none": 1.0},
            }
        })
    )

    primary = primary_metric_view(load_eval_results(tmp_path))

    assert primary.iloc[0]["value"] == 0.75
    assert primary.iloc[0]["run_id"] == "20260102T000000Z_vA_full"


def test_programbench_primary_metric_is_resolved_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    task_dir = tmp_path / "20260101T000000Z_vA_full" / "programbench"
    task_dir.mkdir(parents=True)
    (task_dir / "results_programbench.json").write_text(
        json.dumps({
            "results": {
                "programbench": {
                    "resolved_rate,none": 0.2,
                    "almost_resolved_rate,none": 0.4,
                    "avg_test_pass_rate,none": 0.7,
                }
            }
        })
    )

    full = load_eval_results(tmp_path)
    primary = primary_metric_view(full)

    assert full.iloc[0]["dim"] == "agentic_code"
    assert primary.iloc[0]["metric"] == "resolved_rate,none"
    assert primary.iloc[0]["value"] == 0.2


def test_terminal_bench_primary_metric_is_resolved_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    task_dir = tmp_path / "20260101T000000Z_vA_full" / "terminal_bench"
    task_dir.mkdir(parents=True)
    (task_dir / "results_terminal_bench.json").write_text(
        json.dumps({
            "results": {
                "terminal_bench": {
                    "resolved_rate,none": 0.5,
                    "n_tasks,none": 2,
                }
            }
        })
    )

    full = load_eval_results(tmp_path)
    primary = primary_metric_view(full)

    assert full.iloc[0]["dim"] == "agentic_code"
    assert primary.iloc[0]["metric"] == "resolved_rate,none"
    assert primary.iloc[0]["value"] == 0.5


def test_memory_stability_results_load_as_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "get_registry", lambda: _FakeRegistry())
    task_dir = tmp_path / "20260101T000000Z_vA_full" / "memory_stability"
    task_dir.mkdir(parents=True)
    (task_dir / "results_memory_stability.json").write_text(
        json.dumps(
            {
                "results": {
                    "memory_stability": {
                        "acc,none": 0.25,
                        "episodic_only_acc,none": 1.0,
                    }
                }
            }
        )
    )

    full = load_eval_results(tmp_path)
    primary = primary_metric_view(full)

    assert full.iloc[0]["dim"] == "diagnostic"
    assert primary.iloc[0]["metric"] == "acc,none"
    assert primary.iloc[0]["value"] == 0.25
