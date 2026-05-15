"""Tests for eval suite support gating."""

from __future__ import annotations

from llm_bench.evals.suites import (
    external_suite,
    external_supports_fmt,
    long_suite,
    smoke_suite,
    supports_capabilities,
    supports_fmt,
)


def test_loglikelihood_tasks_need_explicit_logprob_capability():
    assert supports_fmt("hellaswag", "mlx") is False
    assert supports_fmt("hellaswag", "gguf") is False


def test_external_code_runners_are_not_supported_on_mlx_until_server_bug_is_fixed():
    assert external_supports_fmt("humaneval", "evalplus", "mlx") is False
    assert external_supports_fmt("livecodebench", "livecodebench", "mlx") is False
    assert external_supports_fmt("bigcodebench_hard", "bigcodebench", "mlx") is False
    assert external_supports_fmt("humaneval", "evalplus", "gguf") is True
    assert external_supports_fmt("bigcodebench_hard", "bigcodebench", "gguf") is True


def test_sourceqa_external_runner_is_supported_for_both_formats():
    assert ("diagnostic", "sourceqa", "sourceqa") in external_suite()
    assert external_supports_fmt("sourceqa", "sourceqa", "mlx") is True
    assert external_supports_fmt("sourceqa", "sourceqa", "gguf") is True


def test_memory_stability_external_runner_is_supported_for_chat_formats():
    assert ("diagnostic", "memory_stability", "memory_stability") in external_suite()
    assert external_supports_fmt("memory_stability", "memory_stability", "mlx") is True
    assert external_supports_fmt("memory_stability", "memory_stability", "gguf") is True


def test_terminal_bench_external_runner_is_supported_for_chat_formats():
    assert ("agentic_code", "terminal_bench", "terminal_bench") in external_suite()
    assert external_supports_fmt("terminal_bench", "terminal_bench", "mlx") is True
    assert external_supports_fmt("terminal_bench", "terminal_bench", "gguf") is True


def test_fresh_and_korean_external_runners_are_supported_for_chat_formats():
    assert ("fresh", "livebench_subset", "livebench") in external_suite()
    assert ("korean", "kmmlu_pro", "kmmlu_pro") in external_suite()
    assert external_supports_fmt("livebench_subset", "livebench", "mlx") is True
    assert external_supports_fmt("kmmlu_pro", "kmmlu_pro", "gguf") is True


def test_smoke_suite_excludes_heavy_or_superseded_tasks():
    tasks = {task for _, task in smoke_suite()}

    assert "humaneval_instruct" not in tasks
    assert "longbench" not in tasks


def test_long_suite_is_explicit_longbench_only():
    assert long_suite() == [("long", "longbench")]


def test_loglikelihood_support_uses_capabilities_not_format_names():
    assert supports_capabilities("hellaswag", {"chat", "completions"}) is False
    assert supports_capabilities("hellaswag", {"chat", "completions", "logprobs"}) is True
