"""Tests for registry-driven model sync helpers."""

from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_sync_models():
    path = ROOT / "scripts" / "sync_models.py"
    spec = importlib.util.spec_from_file_location("sync_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _Download:
    repo: str
    pattern: str
    revision: str | None = None


def test_hf_download_uses_repo_root_for_nested_patterns(tmp_path, monkeypatch):
    sync_models = _load_sync_models()
    captured = {}

    class SplitVariant:
        key = "split"
        fmt = "gguf"
        resolved_path = str(tmp_path / "root" / "Q4_K_M" / "model-00001-of-00002.gguf")
        download = _Download(repo="org/model-GGUF", pattern="Q4_K_M/*", revision="abc123")

    monkeypatch.setattr(sync_models.shutil, "which", lambda _: "/usr/bin/hf")

    def fake_run(cmd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sync_models.subprocess, "run", fake_run)

    assert sync_models.hf_download(SplitVariant()) == 0
    assert captured["cmd"][captured["cmd"].index("--local-dir") + 1] == str(tmp_path / "root")
    assert captured["cmd"][captured["cmd"].index("--include") + 1] == "Q4_K_M/*"
