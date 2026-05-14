"""Tests for the LiveCodeBench external runner.

Covers the contract that aggregate.py relies on:
  - returns {"task": "livecodebench", "results": {...}, ...}
  - on success, results contain a numeric pass@1 in [0, 1]
  - writes a synthetic results_*.json that aggregate.py can pick up
  - error paths surface clearly (package missing, subprocess timeout, rc!=0)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from llm_bench.evals import livecodebench_runner as lcb


def test_missing_package_returns_error(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVE_CODE_BENCH_REPO", raising=False)
    monkeypatch.setattr(lcb, "_module_exists", lambda _: False)
    res = lcb.run_livecodebench(
        release="release_v6",
        base_url="http://localhost:9090/v1",
        model_label="m",
        output_dir=tmp_path,
    )
    assert res["task"] == "livecodebench"
    assert "error" in res
    assert "lcb_runner" in res["error"] or "LiveCodeBench" in res["error"]


def test_availability_requires_runner_main(monkeypatch):
    monkeypatch.delenv("LIVE_CODE_BENCH_REPO", raising=False)
    seen = {}

    def fake_module_exists(name):
        seen["name"] = name
        return False

    monkeypatch.setattr(lcb, "_module_exists", fake_module_exists)

    assert lcb.livecodebench_available() is False
    assert seen["name"] == "lcb_runner.runner.main"


def test_successful_run_returns_pass_at_1(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_module_exists", lambda _: True)
    monkeypatch.setattr(lcb, "_source_checkout", lambda: None)

    # Pretend lcb_runner wrote a Scores.json with pass@1
    scores_path = tmp_path / "Scores.json"
    scores_path.write_text(json.dumps({"pass@1": 0.213, "pass@5": 0.34}))

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="pass@1: 0.213\nDone.", stderr=""
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(
        release="release_v6", base_url="http://localhost:9090/v1",
        model_label="my-model", output_dir=tmp_path,
    )
    assert res["task"] == "livecodebench"
    assert "error" not in res
    assert res["results"]["livecodebench"]["pass@1,none"] == pytest.approx(0.213)
    # Synthetic results_*.json must exist for aggregate.py to pick up
    synth = list(tmp_path.glob("results_*_livecodebench.json"))
    assert len(synth) == 1
    parsed = json.loads(synth[0].read_text())
    assert parsed["results"]["livecodebench"]["pass@1,none"] == pytest.approx(0.213)


def test_pass_at_1_above_one_normalised(tmp_path, monkeypatch):
    """Some lcb_runner versions report pass@1 as a percent (e.g. 21.3)."""
    monkeypatch.setattr(lcb, "_module_exists", lambda _: True)
    monkeypatch.setattr(lcb, "_source_checkout", lambda: None)
    work_dir = tmp_path / "_lcb_work"
    work_dir.mkdir()
    (work_dir / "Scores.json").write_text(json.dumps({"pass@1": 21.3}))

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(release="release_v6", base_url="http://x/v1",
                                model_label="m", output_dir=tmp_path)
    assert res["results"]["livecodebench"]["pass@1,none"] == pytest.approx(0.213)


def test_stdout_fallback_when_no_json(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_module_exists", lambda _: True)
    monkeypatch.setattr(lcb, "_source_checkout", lambda: None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="...\npass@1 = 0.412\n...", stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(release="release_v6", base_url="http://x/v1",
                                model_label="m", output_dir=tmp_path)
    assert res["results"]["livecodebench"]["pass@1,none"] == pytest.approx(0.412)


def test_extracts_pass_at_1_from_lcb_eval_json(tmp_path):
    eval_path = tmp_path / "output" / "Model" / "Scenario.codegeneration_1_0.2_eval.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps([{"pass@1": 0.25, "detail": {"pass@1": {"0": 0.0}}}]))

    assert lcb._extract_pass_at_1([tmp_path / "missing", tmp_path / "output"], "") == 0.25


def test_extracts_numeric_stdout_score(tmp_path):
    assert lcb._extract_pass_at_1(tmp_path, "Loaded 10 problems\n0.0\n") == 0.0


def test_subprocess_failure_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_module_exists", lambda _: True)
    monkeypatch.setattr(lcb, "_source_checkout", lambda: None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1,
                                            stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(release="release_v6", base_url="http://x/v1",
                                model_label="m", output_dir=tmp_path)
    assert "error" in res
    assert "rc=1" in res["error"]
    assert Path(res["log"]).exists()


def test_subprocess_timeout_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_module_exists", lambda _: True)
    monkeypatch.setattr(lcb, "_source_checkout", lambda: None)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(release="release_v6", base_url="http://x/v1",
                                model_label="m", output_dir=tmp_path, timeout_s=1)
    assert "error" in res
    assert "timeout" in res["error"].lower()


def test_source_checkout_makes_runner_available(tmp_path, monkeypatch):
    checkout = tmp_path / "LiveCodeBench"
    main = checkout / "lcb_runner" / "runner" / "main.py"
    main.parent.mkdir(parents=True)
    main.write_text("")
    monkeypatch.setenv("LIVE_CODE_BENCH_REPO", str(checkout))
    monkeypatch.setattr(lcb, "_module_exists", lambda _: False)

    assert lcb.livecodebench_available() is True


def test_run_uses_source_checkout_pythonpath_and_openai_base(tmp_path, monkeypatch):
    checkout = tmp_path / "LiveCodeBench"
    main = checkout / "lcb_runner" / "runner" / "main.py"
    main.parent.mkdir(parents=True)
    main.write_text("")
    stale_scores = checkout / "output" / "StaleModel" / "Scores.json"
    stale_scores.parent.mkdir(parents=True)
    stale_scores.write_text(json.dumps({"pass@1": 0.99}))
    captured = {}
    monkeypatch.setenv("LIVE_CODE_BENCH_REPO", str(checkout))
    monkeypatch.setenv("LIVE_CODE_BENCH_START_DATE", "2025-03-29")
    monkeypatch.setenv("LIVE_CODE_BENCH_END_DATE", "2025-04-06")
    monkeypatch.setenv("LIVE_CODE_BENCH_MAX_TOKENS", "1024")
    monkeypatch.setenv("LIVE_CODE_BENCH_OPENAI_TIMEOUT", "1200")
    monkeypatch.setenv("LIVE_CODE_BENCH_NOT_FAST", "1")
    monkeypatch.setattr(lcb, "_module_exists", lambda _: False)

    def fake_run(cmd, *, capture_output, text, env, timeout, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="pass@1: 0.5", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = lcb.run_livecodebench(
        release="release_v6",
        base_url="http://localhost:9090/v1",
        model_label="local-variant",
        output_dir=tmp_path / "out",
        limit=1,
    )

    assert res["results"]["livecodebench"]["pass@1,none"] == pytest.approx(0.5)
    assert captured["env"]["OPENAI_BASE_URL"] == "http://localhost:9090/v1"
    assert captured["env"]["OPENAI_KEY"] == "local-no-auth"
    assert str(checkout) in captured["env"]["PYTHONPATH"]
    assert captured["cwd"] == str(tmp_path / "out" / "_lcb_work")
    assert captured["cmd"][captured["cmd"].index("--n") + 1] == "1"
    assert captured["cmd"][captured["cmd"].index("--start_date") + 1] == "2025-03-29"
    assert captured["cmd"][captured["cmd"].index("--end_date") + 1] == "2025-04-06"
    assert captured["cmd"][captured["cmd"].index("--max_tokens") + 1] == "1024"
    assert captured["cmd"][captured["cmd"].index("--openai_timeout") + 1] == "1200"
    assert "--not_fast" in captured["cmd"]
    assert "--debug" in captured["cmd"]
    assert "gpt-4o-mini-2024-07-18" in captured["cmd"]
