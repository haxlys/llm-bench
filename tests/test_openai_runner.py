"""Tests for OpenAI-compatible endpoint speed runner."""

from __future__ import annotations

from llm_bench.runners.base import Scenario
from llm_bench.runners.openai_runner import OpenAICompatibleRunner


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
        }


def test_openai_runner_measures_endpoint_with_usage_tokens(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr("llm_bench.runners.openai_runner.requests.post", fake_post)

    runner = OpenAICompatibleRunner(
        model_id="provider/model",
        model_label="actual/model",
        base_url="https://models.example.test/v1",
        quant="hosted",
        variant_key="hosted-api",
        api_key="secret-token",
    )

    result = runner.run(Scenario("p128_g32", 128, 32), run_idx=1)

    assert calls[0]["url"] == "https://models.example.test/v1/chat/completions"
    assert calls[0]["json"]["model"] == "actual/model"
    assert calls[0]["json"]["max_tokens"] == 32
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert result.fmt == "api"
    assert result.backend == "openai-compatible"
    assert result.artifact_type == "endpoint"
    assert result.n_prompt == 120
    assert result.n_gen == 30
    assert result.pp_tps > 0
    assert result.tg_tps > 0
    assert result.raw["measurement"] == "wall_clock_effective"
