"""Tests for the LiveBench external runner."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from llm_bench.evals import livebench_runner as livebench


def test_missing_checkout_returns_error(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVEBENCH_REPO", raising=False)

    res = livebench.run_livebench(
        base_url="http://localhost:9090/v1",
        model_label="m",
        output_dir=tmp_path,
        limit=1,
    )

    assert res["task"] == "livebench_subset"
    assert "error" in res
    assert "LIVEBENCH_REPO" in res["error"]


def test_successful_run_parses_group_scores(tmp_path, monkeypatch):
    checkout = tmp_path / "LiveBench" / "livebench"
    checkout.mkdir(parents=True)
    (checkout / "run_livebench.py").write_text("")
    monkeypatch.setenv("LIVEBENCH_REPO", str(tmp_path / "LiveBench"))

    captured = {}

    def fake_run(cmd, *, capture_output, text, env, timeout, cwd):
        del capture_output, text, env, timeout
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        with (Path(cwd) / "all_groups.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "category", "score"])
            writer.writeheader()
            writer.writerow({"model": "my_model", "category": "overall", "score": "52.0"})
            writer.writerow({"model": "my_model", "category": "reasoning", "score": "0.5"})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = livebench.run_livebench(
        base_url="http://localhost:9090/v1",
        model_label="my/model",
        output_dir=tmp_path / "out",
        limit=2,
    )

    assert res["task"] == "livebench_subset"
    assert "error" not in res
    assert res["results"]["livebench_subset"]["score,none"] == pytest.approx(0.52)
    assert res["results"]["livebench_subset"]["reasoning_score,none"] == pytest.approx(0.5)
    assert captured["cwd"] == str(checkout)
    assert captured["cmd"][captured["cmd"].index("--question-end") + 1] == "2"
    assert captured["cmd"][captured["cmd"].index("--api-base") + 1] == "http://localhost:9090/v1"

    synth = list((tmp_path / "out").glob("results_*_livebench_subset.json"))
    assert len(synth) == 1
    parsed = json.loads(synth[0].read_text())
    assert parsed["results"]["livebench_subset"]["score,none"] == pytest.approx(0.52)


def test_stdout_score_fallback():
    scores = livebench._parse_livebench_scores([], "Overall: 41.5\n", "m")

    assert scores["score"] == pytest.approx(0.415)


def test_parses_wide_livebench_pivot_csv(tmp_path):
    path = tmp_path / "all_groups.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "average", "math", "reasoning"])
        writer.writeheader()
        writer.writerow({"model": "my_model", "average": "61.2", "math": "53.7", "reasoning": "64.0"})

    scores = livebench._parse_livebench_scores([tmp_path], "", "my_model")

    assert scores["score"] == pytest.approx(0.612)
    assert scores["math_score"] == pytest.approx(0.537)
    assert scores["reasoning_score"] == pytest.approx(0.64)
