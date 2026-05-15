"""Tests for the Terminal-Bench run wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    path = ROOT / "scripts" / "run_terminal_bench.py"
    spec = importlib.util.spec_from_file_location("run_terminal_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_terminal_model_label_defaults_to_litellm_openai_provider():
    mod = _load_script()

    assert mod._terminal_model_label("local-model", None) == "openai/local-model"
    assert mod._terminal_model_label("openai/gpt-5.5", None) == "openai/gpt-5.5"
    assert mod._terminal_model_label("local-model", "openrouter/model") == "openrouter/model"


def test_split_task_ids_accepts_repeated_and_comma_separated_values():
    mod = _load_script()

    assert mod._split_task_ids(("hello-world,git-fix", " package-install ")) == [
        "hello-world",
        "git-fix",
        "package-install",
    ]


def test_effective_n_tasks_defaults_to_one_without_explicit_task_filter():
    mod = _load_script()

    assert mod._effective_n_tasks([], None, False) == 1
    assert mod._effective_n_tasks(["hello-world"], None, False) is None
    assert mod._effective_n_tasks([], 3, False) == 3
    assert mod._effective_n_tasks([], 3, True) is None
