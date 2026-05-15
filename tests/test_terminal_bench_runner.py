"""Tests for Terminal-Bench result ingestion and command construction."""

from __future__ import annotations

import json

import pytest

from llm_bench.evals.terminal_bench_runner import _terminal_bench_command
from llm_bench.evals.terminal_bench_runner import _terminal_bench_env
from llm_bench.evals.terminal_bench_runner import _safe_run_id
from llm_bench.evals.terminal_bench_runner import aggregate_terminal_bench_scores
from llm_bench.evals.terminal_bench_runner import import_terminal_bench_results


def test_import_terminal_bench_results_writes_synthetic_metrics(tmp_path):
    raw_results = tmp_path / "results.json"
    raw_results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "hello-world",
                        "is_resolved": True,
                        "total_input_tokens": 100,
                        "total_output_tokens": 10,
                        "trial_started_at": "2026-05-15T00:00:00+00:00",
                        "trial_ended_at": "2026-05-15T00:00:20+00:00",
                    },
                    {
                        "task_id": "failing-task",
                        "is_resolved": False,
                        "failure_mode": "test_failed",
                        "total_input_tokens": 200,
                        "total_output_tokens": 20,
                        "trial_started_at": "2026-05-15T00:00:00+00:00",
                        "trial_ended_at": "2026-05-15T00:01:00+00:00",
                    },
                ]
            }
        )
    )

    result = import_terminal_bench_results(
        raw_results=raw_results,
        output_dir=tmp_path / "out",
        metadata={"dataset": "terminal-bench-core==0.1.1"},
    )

    metrics = result["results"]["terminal_bench"]
    assert metrics["resolved_rate,none"] == pytest.approx(0.5)
    assert metrics["n_tasks,none"] == 2
    assert metrics["n_resolved,none"] == 1
    assert metrics["avg_input_tokens,none"] == pytest.approx(150.0)
    assert metrics["avg_output_tokens,none"] == pytest.approx(15.0)
    assert metrics["avg_wall_s,none"] == pytest.approx(40.0)

    payload = json.loads(next((tmp_path / "out").glob("results_*_terminal_bench.json")).read_text())
    assert payload["results"]["terminal_bench"] == metrics
    assert payload["samples"][0]["task_id"] == "hello-world"
    assert payload["metadata"]["dataset"] == "terminal-bench-core==0.1.1"


def test_aggregate_terminal_bench_scores_handles_empty_samples():
    assert aggregate_terminal_bench_scores([]) == {
        "resolved_rate,none": 0.0,
        "n_tasks,none": 0,
        "n_resolved,none": 0,
        "n_unresolved,none": 0,
        "avg_input_tokens,none": 0.0,
        "avg_output_tokens,none": 0.0,
        "avg_wall_s,none": 0.0,
    }


def test_terminal_bench_command_includes_core_options(tmp_path):
    cmd = _terminal_bench_command(
        terminal_bench_bin="uvx --from terminal-bench tb",
        dataset="terminal-bench-core==0.1.1",
        output_path=tmp_path / "tb",
        run_id="run-1",
        agent="terminus",
        model_label="openai/local-model",
        task_ids=["hello-world"],
        n_tasks=1,
        global_agent_timeout_sec=600,
        global_test_timeout_sec=300,
        cleanup=False,
    )

    assert cmd == [
        "uvx",
        "--from",
        "terminal-bench",
        "tb",
        "run",
        "--dataset",
        "terminal-bench-core==0.1.1",
        "--output-path",
        str(tmp_path / "tb"),
        "--run-id",
        "run-1",
        "--agent",
        "terminus",
        "--model",
        "openai/local-model",
        "--n-concurrent",
        "1",
        "--no-upload-results",
        "--task-id",
        "hello-world",
        "--n-tasks",
        "1",
        "--global-agent-timeout-sec",
        "600",
        "--global-test-timeout-sec",
        "300",
        "--no-cleanup",
    ]


def test_terminal_bench_env_sets_openai_and_docker(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env = _terminal_bench_env(
        base_url="http://127.0.0.1:9090/v1/",
        api_key=None,
        docker_host="unix:///tmp/docker.sock",
    )

    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9090/v1"
    assert env["OPENAI_API_KEY"] == "sk-local"
    assert env["DOCKER_HOST"] == "unix:///tmp/docker.sock"


def test_safe_run_id_is_docker_compose_project_name_friendly():
    assert _safe_run_id("20260515T045850Z_deepseek-v4-flash-gguf-iq2xxs") == (
        "20260515t045850z_deepseek-v4-flash-gguf-iq2xxs"
    )
    assert _safe_run_id("Run With Spaces!") == "run-with-spaces"
