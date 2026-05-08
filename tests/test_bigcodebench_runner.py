"""Tests for the BigCodeBench-Hard external runner."""

from __future__ import annotations

import json
import subprocess

import pytest

from llm_bench.evals import bigcodebench_runner as bcb


def test_missing_package_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bcb, "_module_exists", lambda _: False)

    res = bcb.run_bigcodebench_hard(
        base_url="http://localhost:9090/v1",
        model_label="m",
        output_dir=tmp_path,
    )

    assert res["task"] == "bigcodebench_hard"
    assert "error" in res
    assert "bigcodebench" in res["error"]


def test_successful_run_parses_pass_at_1(tmp_path, monkeypatch):
    monkeypatch.setattr(bcb, "_module_exists", lambda _: True)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        (tmp_path / "model--bigcodebench-instruct--openai-pass_at_k.json").write_text(
            json.dumps({"pass@1": 42.0})
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = bcb.run_bigcodebench_hard(
        base_url="http://localhost:9090/v1",
        model_label="my-model",
        output_dir=tmp_path,
        limit=3,
    )

    assert res["task"] == "bigcodebench_hard"
    assert "error" not in res
    assert res["results"]["bigcodebench_hard"]["pass@1,none"] == pytest.approx(0.42)
    assert captured["cmd"][captured["cmd"].index("--base_url") + 1] == "http://localhost:9090/v1"
    assert captured["cmd"][captured["cmd"].index("--id_range") + 1] == "0-3"
    assert captured["env"]["OPENAI_API_KEY"] == "local-no-auth"

    synth = list(tmp_path.glob("results_*_bigcodebench_hard.json"))
    assert len(synth) == 1
    parsed = json.loads(synth[0].read_text())
    assert parsed["results"]["bigcodebench_hard"]["pass@1,none"] == pytest.approx(0.42)


def test_stdout_pass_at_1_fallback(tmp_path):
    assert bcb._extract_pass_at_1(tmp_path, "pass@1 = 0.25") == pytest.approx(0.25)
