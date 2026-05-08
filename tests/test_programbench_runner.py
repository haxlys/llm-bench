"""Tests for ProgramBench eval result ingestion."""

from __future__ import annotations

import json

import pytest

from llm_bench.evals.programbench_runner import run_programbench_import
from llm_bench.evals.programbench_runner import score_programbench_eval_file


def _write_eval(path, statuses, *, error_code=None, branch_errors=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "test_results": [
                    {"name": f"test_{idx}", "branch": "b", "status": status, "extra": {}}
                    for idx, status in enumerate(statuses)
                ],
                "error_code": error_code,
                "test_branch_errors": branch_errors or {},
                "warnings": [],
            }
        )
    )


def _write_tests_json(path, *, ignored_tests=None, ignored=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "branches": {
                    "b": {
                        "ignored": ignored,
                        "tests": ["test_0", "test_1", "test_2"],
                        "ignored_tests": ignored_tests or [],
                    }
                }
            }
        )
    )


def test_score_eval_file_marks_resolved_only_when_all_tests_pass(tmp_path):
    eval_json = tmp_path / "org__tool.abc123" / "org__tool.abc123.eval.json"
    _write_eval(eval_json, ["passed", "passed"])

    score = score_programbench_eval_file(eval_json)

    assert score.instance_id == "org__tool.abc123"
    assert score.pass_rate == pytest.approx(1.0)
    assert score.resolved is True
    assert score.almost_resolved is True


def test_score_eval_file_filters_ignored_tests_from_tasks_dir(tmp_path):
    eval_json = tmp_path / "run" / "org__tool.abc123" / "org__tool.abc123.eval.json"
    _write_eval(eval_json, ["passed", "failure"])
    tasks_dir = tmp_path / "tasks"
    _write_tests_json(
        tasks_dir / "org__tool.abc123" / "tests.json",
        ignored_tests=[{"name": "test_1"}],
    )

    score = score_programbench_eval_file(eval_json, tasks_dir=tasks_dir)

    assert score.n_tests == 1
    assert score.n_passed == 1
    assert score.resolved is True


def test_programbench_import_writes_synthetic_results(tmp_path):
    source = tmp_path / "programbench-run"
    _write_eval(source / "solved" / "solved.eval.json", ["passed", "passed"])
    _write_eval(
        source / "almost" / "almost.eval.json",
        ["passed"] * 19 + ["failure"],
    )
    _write_eval(source / "compile-error" / "compile-error.eval.json", [], error_code="compile_failed")

    output = tmp_path / "out"
    res = run_programbench_import(source_dir=source, output_dir=output)

    metrics = res["results"]["programbench"]
    assert metrics["resolved_rate,none"] == pytest.approx(1 / 3, abs=0.0001)
    assert metrics["almost_resolved_rate,none"] == pytest.approx(2 / 3, abs=0.0001)
    assert metrics["avg_test_pass_rate,none"] == pytest.approx(0.65)
    assert metrics["n_instances,none"] == 3
    assert metrics["n_resolved,none"] == 1
    assert metrics["n_almost_resolved,none"] == 2
    assert metrics["n_tests,none"] == 22
    assert metrics["n_passed_tests,none"] == 21
    assert metrics["n_system_error_instances,none"] == 1

    written = list(output.glob("results_*_programbench.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["results"]["programbench"] == metrics
    assert len(payload["samples"]) == 3


def test_programbench_import_returns_error_when_no_eval_files(tmp_path):
    res = run_programbench_import(source_dir=tmp_path, output_dir=tmp_path / "out")

    assert res["task"] == "programbench"
    assert "error" in res
