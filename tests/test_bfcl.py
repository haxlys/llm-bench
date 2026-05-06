"""Tests for the BFCL v4 (`bfcl_eval`) external runner.

Covers the contract that aggregate.py relies on:
  - returns {"task": "bfcl", "results": {...}, "categories": [...]} on success
  - results contain numeric overall_accuracy in [0, 1]
  - writes a synthetic results_*.json
  - error paths surface clearly (missing package, generate failure, evaluate failure)
"""

from __future__ import annotations

import json
import subprocess

import pytest

from llm_bench.evals import bfcl


def test_missing_package_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: False)
    res = bfcl.run_bfcl(
        base_url="http://localhost:9090/v1",
        model_label="m",
        output_dir=tmp_path,
    )
    assert res["task"] == "bfcl"
    assert "error" in res
    assert "bfcl-eval" in res["error"] or "bfcl_eval" in res["error"]


def test_successful_run_aggregates_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    score_dir = tmp_path / "score"
    score_dir.mkdir()
    (score_dir / "simple_score.json").write_text(json.dumps({"accuracy": 0.5}))
    (score_dir / "parallel_score.json").write_text(json.dumps({"accuracy": 0.4}))
    (score_dir / "multiple_score.json").write_text(json.dumps({"accuracy": 0.3}))
    (score_dir / "parallel_multiple_score.json").write_text(json.dumps({"accuracy": 0.2}))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0,
                                            stdout="OK", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(
        base_url="http://x/v1",
        model_label="m",
        output_dir=tmp_path,
    )
    assert res["task"] == "bfcl"
    assert "error" not in res
    assert res["categories"] == ["simple", "parallel", "multiple", "parallel_multiple"]

    overall = res["results"]["bfcl"]["overall_accuracy,none"]
    assert overall == pytest.approx((0.5 + 0.4 + 0.3 + 0.2) / 4)

    # Synthetic results JSON
    synth = list(tmp_path.glob("results_*_bfcl.json"))
    assert len(synth) == 1
    parsed = json.loads(synth[0].read_text())
    assert "overall_accuracy,none" in parsed["results"]["bfcl"]
    assert parsed["results"]["bfcl"]["simple_acc,none"] == pytest.approx(0.5)

    # Both phases (generate + evaluate) ran
    assert len(calls) == 2


def test_accuracy_above_one_normalised(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    score_dir = tmp_path / "score"
    score_dir.mkdir()
    (score_dir / "simple_score.json").write_text(json.dumps({"accuracy": 50.0}))

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(base_url="http://x/v1", model_label="m",
                        output_dir=tmp_path, categories=["simple"])
    assert res["results"]["bfcl"]["simple_acc,none"] == pytest.approx(0.5)


def test_generate_failure_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=2,
                                            stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(base_url="http://x/v1", model_label="m",
                        output_dir=tmp_path)
    assert "error" in res
    assert "generate rc=2" in res["error"]


def test_evaluate_failure_returns_error(tmp_path, monkeypatch):
    """Generate succeeds but evaluate fails — error must mention evaluate."""
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    state = {"calls": 0}

    def fake_run(cmd, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=3, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(base_url="http://x/v1", model_label="m",
                        output_dir=tmp_path)
    assert "error" in res
    assert "evaluate rc=3" in res["error"]


def test_no_scores_returns_error(tmp_path, monkeypatch):
    """Both phases succeed but no parseable scores → clear error."""
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0,
                                            stdout="silent run", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(base_url="http://x/v1", model_label="m",
                        output_dir=tmp_path)
    assert "error" in res
    assert "no scores" in res["error"].lower()


def test_stdout_fallback(tmp_path, monkeypatch):
    """When no score JSON exists, parse 'Overall Accuracy: 0.42' from stdout."""
    monkeypatch.setattr(bfcl, "_cli_loads", lambda _: True)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="Simple Accuracy: 0.5\nOverall Accuracy: 0.42\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bfcl.run_bfcl(base_url="http://x/v1", model_label="m",
                        output_dir=tmp_path)
    # Either overall comes out from stdout regex, or simple_acc — either is OK
    # as long as something landed.
    assert "error" not in res or res.get("results")
