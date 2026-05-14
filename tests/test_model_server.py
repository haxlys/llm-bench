"""Tests for eval server lifecycle adapters."""

from __future__ import annotations

import subprocess
import json

import llm_bench.evals.server as server_mod
from llm_bench.evals.server import ModelServer


def test_endpoint_model_server_returns_existing_base_url_without_spawning(monkeypatch):
    def fail_popen(*args, **kwargs):
        raise AssertionError("endpoint adapters must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    with ModelServer(
        fmt="api",
        backend="openai-compatible",
        artifact_type="endpoint",
        model_path="https://models.example.test/v1",
    ) as base_url:
        assert base_url == "https://models.example.test/v1"


def test_endpoint_model_server_appends_v1_when_missing():
    server = ModelServer(
        fmt="api",
        backend="openai-compatible",
        artifact_type="endpoint",
        model_path="https://models.example.test/",
    )

    assert server.base_url == "https://models.example.test/v1"


def test_gguf_model_server_uses_configured_context_size(monkeypatch):
    monkeypatch.setattr(server_mod.shutil, "which", lambda _name: "/opt/homebrew/bin/llama-server")

    server = ModelServer(
        fmt="gguf",
        backend="gguf",
        artifact_type="gguf_file",
        model_path="/models/model.gguf",
        context_size=65536,
    )

    cmd = server._build_cmd()

    assert cmd[cmd.index("-c") + 1] == "65536"
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]) == {
        "enable_thinking": False,
    }


def test_mlx_model_server_disables_thinking_by_default():
    server = ModelServer(fmt="mlx", model_path="org/model")

    cmd = server._build_cmd()

    assert "--chat-template-args" in cmd
    raw_args = cmd[cmd.index("--chat-template-args") + 1]
    assert json.loads(raw_args) == {"enable_thinking": False}


def test_mlx_model_server_accepts_custom_chat_template_args():
    server = ModelServer(
        fmt="mlx",
        model_path="org/model",
        chat_template_args={"enable_thinking": True, "preserve_thinking": True},
    )

    cmd = server._build_cmd()

    raw_args = cmd[cmd.index("--chat-template-args") + 1]
    assert json.loads(raw_args) == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
