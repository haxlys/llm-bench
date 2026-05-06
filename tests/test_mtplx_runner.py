"""Tests for the MTPLX speed runner."""

from __future__ import annotations

import json

from llm_bench.runners.base import Scenario
from llm_bench.runners.mtplx_runner import MTPLXRunner


def test_mtplx_runner_parses_mtp_stats(monkeypatch):
    calls = []

    def fake_run_with_time(cmd, env=None, timeout_s=0, check=True):
        calls.append(cmd)
        assert check is False
        payload = {
            "stats": {
                "tok_s": 36.75,
                "generated_tokens": 164,
                "generation_mode": "mtp",
                "verify_ms_per_call": 50.5,
                "verify_calls": 71,
                "accepted_by_depth": [50, 29, 13],
                "drafted_by_depth": [71, 71, 71],
            }
        }
        return json.dumps(payload), "", 8.125, 18.25

    monkeypatch.setattr("llm_bench.runners.mtplx_runner.run_with_time", fake_run_with_time)

    runner = MTPLXRunner(
        model_id="qwen-3.6-27B-MTPLX",
        model_path="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        quant="MLX-4bit",
        variant_key="qwen-3.6-27b-mtplx-speed-mtplx-mtp",
        generation_mode="mtp",
    )

    result = runner.run(Scenario("p256_g128", 256, 128), run_idx=1)

    assert "--mtp" in calls[0]
    assert result.backend == "mtplx"
    assert result.artifact_type == "hf_repo"
    assert result.fmt == "mlx"
    assert result.n_gen == 164
    assert result.tg_tps == 36.75
    assert result.raw["generation_mode"] == "mtp"
    assert result.raw["acceptance_rates_by_depth"] == [0.704225, 0.408451, 0.183099]


def test_mtplx_runner_supports_target_only_ar(monkeypatch):
    def fake_run_with_time(cmd, env=None, timeout_s=0, check=True):
        assert check is False
        payload = {
            "stats": {
                "tok_s": 27.5,
                "generated_tokens": 129,
                "generation_mode": "ar",
                "verify_calls": 0,
                "accepted_by_depth": [],
                "drafted_by_depth": [],
            }
        }
        return json.dumps(payload), "", 8.0, 17.0

    monkeypatch.setattr("llm_bench.runners.mtplx_runner.run_with_time", fake_run_with_time)

    runner = MTPLXRunner(
        model_id="qwen-3.6-27B-MTPLX",
        model_path="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        quant="MLX-4bit",
        variant_key="qwen-3.6-27b-mtplx-speed-mtplx-ar",
        generation_mode="ar",
    )

    result = runner.run(Scenario("p256_g128", 256, 128), run_idx=1)

    assert result.raw["generation_mode"] == "ar"
    assert result.raw["speedup_baseline"] == "target_only_ar"
    assert result.tg_tps == 27.5
