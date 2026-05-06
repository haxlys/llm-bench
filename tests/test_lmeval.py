"""Tests for lm-eval wrapper configuration."""

from __future__ import annotations

import subprocess

import llm_bench.evals.lmeval as lmeval


def test_run_lmeval_sets_openai_api_key(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="gsm8k_cot_zeroshot",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
        api_key="secret-token",
    )

    assert captured["env"]["OPENAI_API_KEY"] == "secret-token"


def test_longbench_uses_shorter_generation_cap(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="longbench",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
    )

    gen_index = captured["cmd"].index("--gen_kwargs") + 1

    assert captured["cmd"][gen_index] == "max_gen_toks=1024"


def test_hrm8k_uses_moderate_generation_cap(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="hrm8k",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
        limit=20,
    )

    gen_index = captured["cmd"].index("--gen_kwargs") + 1

    assert captured["cmd"][gen_index] == "max_gen_toks=1024"


def test_gsm8k_uses_moderate_generation_cap(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="gsm8k_cot_zeroshot",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
        limit=20,
    )

    gen_index = captured["cmd"].index("--gen_kwargs") + 1

    assert captured["cmd"][gen_index] == "max_gen_toks=1024"


def test_ifeval_uses_bounded_instruction_generation_cap(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="leaderboard_ifeval",
        base_url="https://models.example.test/v1",
        model_label="provider/model",
        output_dir=tmp_path,
        limit=20,
    )

    gen_index = captured["cmd"].index("--gen_kwargs") + 1

    assert captured["cmd"][gen_index] == "max_gen_toks=4096"


def test_completion_tasks_can_use_explicit_tokenizer(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, *, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lmeval.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lmeval, "_module_exists", lambda _name: True)
    monkeypatch.setattr(lmeval.subprocess, "run", fake_run)

    lmeval.run_lmeval(
        task="hellaswag",
        base_url="https://models.example.test/v1",
        model_label="local-gguf-alias",
        tokenizer_label="provider/tokenizer-repo",
        output_dir=tmp_path,
        use_chat=False,
    )

    model_args = captured["cmd"][captured["cmd"].index("--model_args") + 1]

    assert "model=local-gguf-alias" in model_args
    assert "tokenizer=provider/tokenizer-repo" in model_args
