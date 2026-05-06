"""Tests for EvalPlus runner command construction."""

from __future__ import annotations

import json
import subprocess

import llm_bench.evals.evalplus_runner as evalplus_runner


def test_run_evalplus_applies_limit_as_id_range(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(evalplus_runner.subprocess, "run", fake_run)

    evalplus_runner.run_evalplus(
        dataset="humaneval",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
        limit=3,
    )

    id_range_index = captured["cmd"].index("--id_range=[0,3]")

    assert id_range_index > 0


def test_find_samples_accepts_evalplus_sanitized_jsonl(tmp_path):
    raw = tmp_path / "humaneval" / "model_openai_temp_0.0.raw.jsonl"
    sample = tmp_path / "humaneval" / "model_openai_temp_0.0.jsonl"
    sample.parent.mkdir()
    raw.write_text("{}\n")
    sample.write_text("{}\n")

    assert evalplus_runner._find_latest_samples(tmp_path) == sample


def test_evalplus_evaluate_disables_memory_limit(monkeypatch, tmp_path):
    sample = tmp_path / "humaneval" / "model_openai_temp_0.0.jsonl"
    sample.parent.mkdir()
    sample.write_text("{}\n")
    captured_envs = []

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured_envs.append(env)
        if "evalplus.evaluate" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(evalplus_runner.subprocess, "run", fake_run)

    evalplus_runner.run_evalplus(
        dataset="humaneval",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
    )

    assert captured_envs[-1]["EVALPLUS_MAX_MEMORY_BYTES"] == "-1"


def test_write_synthetic_results_uses_aggregate_shape(tmp_path):
    path = evalplus_runner._write_synthetic_results(
        tmp_path,
        "humaneval",
        pass_at_1=0.5,
        pass_at_1_plus=0.25,
    )

    assert path.name.startswith("results_")
    payload = json.loads(path.read_text())
    assert payload["results"]["humaneval"]["pass_at_1,base"] == 0.5
    assert payload["results"]["humaneval"]["pass_at_1,plus"] == 0.25
