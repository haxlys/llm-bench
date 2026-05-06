"""OpenAI-compatible endpoint speed runner."""

from __future__ import annotations

import time

import requests

from llm_bench import BENCH_VERSION
from llm_bench.runners.base import BenchResult, Scenario, now_iso


def _filler_prompt(n_prompt: int) -> str:
    """Approximate a prompt of n_prompt tokens without requiring a tokenizer."""
    base = "The quick brown fox jumps over the lazy dog. "
    word_target = max(1, int(n_prompt / 1.3))
    words_per_base = len(base.split())
    repeats = max(1, word_target // words_per_base + 1)
    return (base * repeats).strip()


class OpenAICompatibleRunner:
    """Measure wall-clock effective throughput for an existing /v1 endpoint.

    Hosted APIs usually do not expose separate prefill/generation timings, so
    pp_tps/tg_tps are effective token rates over the full request wall time.
    Raw usage/timing metadata is preserved for downstream interpretation.
    """

    def __init__(
        self,
        model_id: str,
        model_label: str,
        base_url: str,
        quant: str = "hosted",
        variant_key: str = "",
        api_key: str | None = None,
        timeout_s: int = 10 * 60,
        fmt: str = "api",
        backend: str = "openai-compatible",
        artifact_type: str = "endpoint",
    ):
        self.model_id = model_id
        self.model_label = model_label
        self.base_url = _normalize_base_url(base_url)
        self.quant = quant
        self.variant_key = variant_key
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.fmt = fmt
        self.backend = backend
        self.artifact_type = artifact_type

    def run(self, scenario: Scenario, run_idx: int) -> BenchResult:
        prompt = _filler_prompt(scenario.n_prompt)
        payload = {
            "model": self.model_label,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": scenario.n_gen,
        }

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=_auth_headers(self.api_key),
            timeout=self.timeout_s,
        )
        wall = max(time.perf_counter() - t0, 1e-9)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        content = _message_content(data)
        prompt_tokens = int(usage.get("prompt_tokens") or scenario.n_prompt)
        completion_tokens = int(
            usage.get("completion_tokens") or max(1, len(content.split()))
        )

        return BenchResult(
            model_id=self.model_id,
            fmt=self.fmt,
            quant=self.quant,
            scenario=scenario.name,
            n_prompt=prompt_tokens,
            n_gen=completion_tokens,
            pp_tps=round(prompt_tokens / wall, 3),
            tg_tps=round(completion_tokens / wall, 3),
            peak_mem_gb=0.0,
            wall_s=round(wall, 3),
            run_idx=run_idx,
            ts=now_iso(),
            bench_version=BENCH_VERSION,
            variant_key=self.variant_key,
            backend=self.backend,
            artifact_type=self.artifact_type,
            generation_mode="api",
            raw={
                "measurement": "wall_clock_effective",
                "usage": usage,
                "response_chars": len(content),
            },
        )


def _normalize_base_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _message_content(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = first.get("text")
    return text if isinstance(text, str) else ""
