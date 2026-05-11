"""Tests for run_evals orchestration helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_run_evals():
    path = ROOT / "scripts" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_task_result_records_runner_status_and_artifacts(tmp_path):
    run_evals = _load_run_evals()
    trace_path = tmp_path / "trace.jsonl"

    run_evals._trace_task_result(
        trace_path=trace_path,
        variant_key="26B-MoE-gguf-q8",
        task="sourceqa",
        runner="sourceqa",
        dim="source_grounding",
        wall_s=1.23,
        result={
            "task": "sourceqa",
            "results_file": "/tmp/results.json",
            "samples_file": "/tmp/results.json",
            "results": {"sourceqa": {"acc,none": 1.0}},
        },
    )

    row = json.loads(trace_path.read_text().strip())

    assert row["variant"] == "26B-MoE-gguf-q8"
    assert row["task"] == "sourceqa"
    assert row["runner"] == "sourceqa"
    assert row["dim"] == "source_grounding"
    assert row["status"] == "ok"
    assert row["results_file"] == "/tmp/results.json"
    assert row["samples_file"] == "/tmp/results.json"


def test_api_model_label_uses_endpoint_model_id_not_url():
    run_evals = _load_run_evals()

    class EndpointVariant:
        backend = "openai-compatible"
        fmt = "api"
        path = "https://models.example.test/v1"
        model_id = "provider/model"
        api_model = ""
        key = "hosted"

    class ExplicitEndpointVariant(EndpointVariant):
        api_model = "actual/provider-model"

    assert run_evals._api_model_label(EndpointVariant()) == "provider/model"
    assert run_evals._api_model_label(ExplicitEndpointVariant()) == "actual/provider-model"


def test_api_key_from_variant_env(monkeypatch):
    run_evals = _load_run_evals()
    monkeypatch.setenv("PROVIDER_API_KEY", "secret-token")

    class EndpointVariant:
        api_key_env = "PROVIDER_API_KEY"

    assert run_evals._api_key(EndpointVariant()) == "secret-token"


def test_server_context_size_expands_for_longbench():
    run_evals = _load_run_evals()

    assert run_evals._server_context_size([("long", "longbench")]) == 65536
    assert run_evals._server_context_size([("reasoning", "gsm8k_cot_zeroshot")]) == 16384


def test_tokenizer_label_prefers_variant_tokenizer():
    run_evals = _load_run_evals()

    class GGUFVariant:
        tokenizer = "provider/tokenizer"
        api_model = ""
        model_id = "logical-model"
        key = "local-key"
        backend = "gguf"
        fmt = "gguf"

    assert run_evals._tokenizer_label(GGUFVariant()) == "provider/tokenizer"


def test_evalplus_is_skipped_for_limited_matrix():
    run_evals = _load_run_evals()

    assert run_evals._external_skip_reason("evalplus", 1) == "skipped_limit_incompatible"
    assert run_evals._external_skip_reason("evalplus", None) is None


def test_livecodebench_is_skipped_when_runner_unavailable(monkeypatch):
    run_evals = _load_run_evals()
    monkeypatch.setattr(run_evals, "livecodebench_available", lambda: False)

    assert run_evals._external_skip_reason("livecodebench", 1) == "skipped_unavailable_external"


def test_livecodebench_release_defaults_to_constant(monkeypatch):
    run_evals = _load_run_evals()
    monkeypatch.delenv("LIVE_CODE_BENCH_RELEASE", raising=False)
    assert run_evals._livecodebench_release() == run_evals.LCB_RELEASE


def test_livecodebench_release_reads_env_override(monkeypatch):
    run_evals = _load_run_evals()
    monkeypatch.setenv("LIVE_CODE_BENCH_RELEASE", "release_v9")
    assert run_evals._livecodebench_release() == "release_v9"


def test_coverage_summary_marks_status_completion():
    run_evals = _load_run_evals()
    coverage = [
        run_evals._build_coverage_row("reasoning", "gsm8k_cot_zeroshot", "lm-eval", "completed"),
        run_evals._build_coverage_row("korean", "hrm8k", "lm-eval", "failed"),
        run_evals._build_coverage_row("reasoning", "toxigen", "livebench", "skipped_unavailable_external"),
    ]
    run_evals._set_coverage_status(coverage, "korean", "hrm8k", "lm-eval", "completed")
    summary = run_evals._coverage_summary(coverage)
    assert summary["required"] == 3
    assert summary["completed"] == 2
    assert summary["missing_count"] == 1
    assert summary["missing"][0]["task"] == "toxigen"


def test_resilient_ifeval_coverage_uses_resilient_runner():
    run_evals = _load_run_evals()
    coverage = [
        run_evals._build_coverage_row(
            "instruction",
            "leaderboard_ifeval",
            run_evals._lmeval_runner_for_task("leaderboard_ifeval", True),
            "pending",
        )
    ]

    run_evals._set_coverage_status(
        coverage,
        "instruction",
        "leaderboard_ifeval",
        "ifeval_resilient",
        "completed",
    )

    assert coverage == [
        {
            "dim": "instruction",
            "task": "leaderboard_ifeval",
            "runner": "ifeval_resilient",
            "lane": "primary",
            "required": True,
            "status": "completed",
        }
    ]


def test_optional_rows_do_not_fail_strict_coverage():
    run_evals = _load_run_evals()
    coverage = [
        run_evals._build_coverage_row("code", "humaneval", "evalplus", "completed"),
        run_evals._build_coverage_row(
            "tool",
            "bfcl",
            "bfcl",
            "skipped_optional_disabled",
        ),
    ]

    summary = run_evals._coverage_summary(coverage)

    assert summary["required"] == 1
    assert summary["completed"] == 1
    assert summary["missing_count"] == 0
    assert summary["optional"] == 1


def test_frontier_runners_are_skipped_when_unavailable(monkeypatch):
    run_evals = _load_run_evals()
    monkeypatch.setattr(run_evals, "bigcodebench_available", lambda: False)
    monkeypatch.setattr(run_evals, "livebench_available", lambda: False)
    monkeypatch.setattr(run_evals, "kmmlu_pro_available", lambda: False)

    assert run_evals._external_skip_reason("bigcodebench", 1) == "skipped_unavailable_external"
    assert run_evals._external_skip_reason("livebench", 1) == "skipped_unavailable_external"
    assert run_evals._external_skip_reason("kmmlu_pro", 1) == "skipped_unavailable_external"
