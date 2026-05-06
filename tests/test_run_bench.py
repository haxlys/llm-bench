"""Tests for run_bench target selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from llm_bench.runners.base import Scenario

ROOT = Path(__file__).resolve().parent.parent


def _load_run_bench():
    path = ROOT / "scripts" / "run_bench.py"
    spec = importlib.util.spec_from_file_location("run_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeVariant:
    key = "vA"
    fmt = "mlx"
    backend = "mlx"
    quant = "MLX-8bit"

    def exists_locally(self) -> bool:
        return True


class _FakeRegistry:
    variants = [_FakeVariant()]

    def variant(self, key: str):
        assert key == "vA"
        return self.variants[0]


def test_all_pending_honors_requested_run_count(tmp_path, monkeypatch):
    run_bench = _load_run_bench()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(1, 4):
        (raw_dir / f"r{i}.json").write_text(json.dumps({
            "variant_key": "vA",
            "scenario": "p256_g128",
            "bench_version": "0.3",
            "run_idx": i,
            "ts": f"t{i}",
        }))

    monkeypatch.setattr(run_bench, "RAW_DIR", raw_dir)
    monkeypatch.setattr(run_bench, "get_registry", lambda: _FakeRegistry())

    targets = run_bench._resolve_targets(
        variant_keys=(),
        all_pending=True,
        scenarios=[Scenario("p256_g128", 256, 128)],
        n_required=4,
    )

    assert [t.key for t in targets] == ["vA"]


def test_build_runner_rejects_unknown_backend():
    run_bench = _load_run_bench()

    class UnknownBackendVariant:
        key = "remote"
        backend = "bedrock"
        fmt = "api"
        artifact_type = "endpoint"
        quant = "hosted"
        model_id = "m"
        resolved_path = "https://example.invalid/v1"

    try:
        run_bench._build_runner(UnknownBackendVariant())
    except ValueError as exc:
        assert "No speed runner adapter" in str(exc)
    else:
        raise AssertionError("unknown backend should not fall through to GGUFRunner")


def test_build_runner_supports_openai_compatible_endpoint(monkeypatch):
    run_bench = _load_run_bench()
    monkeypatch.setenv("PROVIDER_API_KEY", "secret-token")

    class EndpointVariant:
        key = "hosted"
        backend = "openai-compatible"
        artifact_type = "endpoint"
        fmt = "api"
        quant = "hosted"
        model_id = "provider/model"
        api_model_label = "actual/model"
        api_key_env = "PROVIDER_API_KEY"
        resolved_path = "https://models.example.test/v1"

    runner = run_bench._build_runner(EndpointVariant())

    assert runner.model_label == "actual/model"
    assert runner.base_url == "https://models.example.test/v1"
    assert runner.api_key == "secret-token"


def test_build_runner_supports_mtplx_backend():
    run_bench = _load_run_bench()

    class MtplxVariant:
        key = "mtplx-mtp"
        backend = "mtplx"
        artifact_type = "hf_repo"
        fmt = "mlx"
        quant = "MLX-4bit"
        model_id = "qwen-3.6-27B-MTPLX"
        resolved_path = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
        notes = "generation_mode=mtp"

    runner = run_bench._build_runner(MtplxVariant())

    assert runner.generation_mode == "mtp"
    assert runner.model_path == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
