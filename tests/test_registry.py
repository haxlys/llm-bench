"""Tests for models/registry.yaml loading + validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.registry import Registry, get_registry, load_registry


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    return p


def test_real_registry_loads():
    """The shipped registry should load and contain expected variants."""
    r = get_registry()
    assert isinstance(r, Registry)
    assert len(r.variants) >= 6
    keys = {v.key for v in r.variants}
    assert "26B-MoE-mlx-8bit" in keys
    assert "31B-Dense-gguf-q8" in keys
    assert "qwen-3.6-27b-mtplx-speed-mlx-4bit" in keys
    assert "qwen-3.6-27b-mtplx-optimized-mlx-mixed4" in keys
    assert "qwen-3.6-27b-mtplx-speed-mtplx-mtp" in keys
    assert "qwen-3.6-27b-mtplx-speed-mtplx-ar" in keys


def test_variant_lookup_by_model_and_tier():
    r = get_registry()
    moe = r.variants_by_model("gemma-4-26B-A4B-it")
    assert len(moe) == 4  # 8bit + 4bit × mlx + gguf
    fourbit = r.variants_by_tier("4bit")
    assert all(v.tier == "4bit" for v in fourbit)


def test_variant_unknown_key_raises():
    r = get_registry()
    with pytest.raises(KeyError):
        r.variant("does-not-exist")


def test_duplicate_keys_rejected(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - {key: dup, fmt: mlx, path: foo, quant: q, tier: 8bit}
  - id: m2
    family: f
    architecture: dense
    variants:
      - {key: dup, fmt: mlx, path: bar, quant: q, tier: 8bit}
"""
    with pytest.raises(ValueError, match="duplicate variant key"):
        load_registry(_write(tmp_path, body))


def test_invalid_fmt_rejected(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - {key: bad, fmt: MLX, path: foo, quant: q, tier: 8bit}
"""
    with pytest.raises(ValueError, match="fmt='MLX'"):
        load_registry(_write(tmp_path, body))


def test_invalid_tier_rejected(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - {key: bad, fmt: mlx, path: foo, quant: q, tier: Q4}
"""
    with pytest.raises(ValueError, match="tier='Q4'"):
        load_registry(_write(tmp_path, body))


def test_invalid_architecture_rejected(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: sparse
    variants:
      - {key: bad, fmt: mlx, path: foo, quant: q, tier: 8bit}
"""
    with pytest.raises(ValueError, match="architecture='sparse'"):
        load_registry(_write(tmp_path, body))


def test_relative_gguf_path_without_download_rejected(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - {key: bad, fmt: gguf, path: relative/file.gguf, quant: Q8_0, tier: 8bit}
"""
    with pytest.raises(ValueError, match="relative"):
        load_registry(_write(tmp_path, body))


def test_default_interpolation(tmp_path):
    body = """
defaults:
  gguf_dir: /tmp/test-models
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - key: ok
        fmt: gguf
        path: "{gguf_dir}/foo.gguf"
        quant: Q8_0
        tier: 8bit
        download: {repo: org/m1, pattern: "*Q8_0*.gguf"}
"""
    r = load_registry(_write(tmp_path, body))
    assert r.variant("ok").resolved_path == "/tmp/test-models/foo.gguf"


def test_unpinned_variant_warns(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - {key: unpinned, fmt: mlx, path: org/m1, quant: MLX-8bit, tier: 8bit}
"""
    with pytest.warns(UserWarning, match="no pinned revision"):
        load_registry(_write(tmp_path, body))


def test_pinned_variant_silent(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - key: pinned
        fmt: mlx
        path: org/m1
        quant: MLX-8bit
        tier: 8bit
        download: {repo: org/m1, revision: abc1234}
"""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        load_registry(_write(tmp_path, body))  # would raise on warn


def test_variant_metadata_inherits_from_model(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: gemma
    architecture: moe
    params_total_b: 26
    params_active_b: 4
    variants:
      - {key: v1, fmt: mlx, path: org/m1, quant: MLX-8bit, tier: 8bit}
"""
    r = load_registry(_write(tmp_path, body))
    v = r.variant("v1")
    assert v.family == "gemma"
    assert v.architecture == "moe"
    assert v.params_total_b == 26
    assert v.params_active_b == 4


def test_variant_can_define_tokenizer_repo(tmp_path):
    body = """
defaults: {}
models:
  - id: m1
    family: qwen
    architecture: dense
    variants:
      - key: v1
        fmt: gguf
        path: /tmp/model.gguf
        quant: Q8_0
        tier: 8bit
        tokenizer: Qwen/Qwen3.5-4B
"""
    r = load_registry(_write(tmp_path, body))

    assert r.variant("v1").tokenizer == "Qwen/Qwen3.5-4B"


def test_variant_defaults_backend_artifact_and_capabilities_from_fmt(tmp_path):
    body = """
defaults:
  gguf_dir: /tmp/test-models
models:
  - id: m1
    family: f
    architecture: dense
    variants:
      - key: mlx-v
        fmt: mlx
        path: org/m1
        quant: MLX-8bit
        tier: 8bit
        download: {repo: org/m1, revision: abc123}
      - key: gguf-v
        fmt: gguf
        path: "{gguf_dir}/m1.gguf"
        quant: Q8_0
        tier: 8bit
        download: {repo: org/m1-gguf, revision: def456}
"""
    r = load_registry(_write(tmp_path, body))

    mlx = r.variant("mlx-v")
    assert mlx.backend == "mlx"
    assert mlx.artifact_type == "hf_repo"
    assert "chat" in mlx.capabilities
    assert "logprobs" not in mlx.capabilities

    gguf = r.variant("gguf-v")
    assert gguf.backend == "gguf"
    assert gguf.artifact_type == "gguf_file"
    assert "chat" in gguf.capabilities
    assert "logprobs" not in gguf.capabilities


def test_generic_endpoint_variant_can_be_declared_without_local_artifact(tmp_path):
    body = """
defaults: {}
models:
  - id: hosted-model
    family: hosted
    architecture: dense
    variants:
      - key: hosted-api
        fmt: api
        backend: openai-compatible
        artifact_type: endpoint
        path: https://example.invalid/v1
        api_model: provider/model-name
        api_key_env: PROVIDER_API_KEY
        quant: hosted
        tier: hosted
        capabilities: [chat, completions, logprobs, tool_calls]
"""
    r = load_registry(_write(tmp_path, body))
    v = r.variant("hosted-api")

    assert v.backend == "openai-compatible"
    assert v.artifact_type == "endpoint"
    assert v.requires_local_artifact is False
    assert v.exists_locally() is True
    assert r.variants_by_backend("openai-compatible") == [v]
    assert "tool_calls" in v.capabilities
    assert v.api_model == "provider/model-name"
    assert v.api_model_label == "provider/model-name"
    assert v.api_key_env == "PROVIDER_API_KEY"


def test_split_gguf_requires_every_shard(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    body = f"""
defaults: {{}}
models:
  - id: split
    family: f
    architecture: dense
    variants:
      - key: split-v
        fmt: gguf
        path: {first}
        quant: Q4_K_M
        tier: 4bit
        download: {{repo: org/split, revision: abc123}}
"""
    v = load_registry(_write(tmp_path, body)).variant("split-v")

    assert v.exists_locally() is False
    second.write_bytes(b"second")
    assert v.exists_locally() is True


def test_mlx_exists_locally_requires_weight_file(tmp_path, monkeypatch):
    """A cached config.json alone is not enough for a benchmarkable MLX model."""
    from llm_bench.registry import Variant
    import llm_bench.registry as registry_mod

    monkeypatch.setattr(registry_mod.Path, "home", lambda: tmp_path)

    repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--org--m"
    snapshot = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    variant = Variant(
        key="v1",
        model_id="m1",
        family="f",
        architecture="dense",
        fmt="mlx",
        path="org/m",
        quant="MLX-8bit",
        tier="8bit",
    )

    assert variant.exists_locally() is False

    (snapshot / "model-00001-of-00002.safetensors").write_text("weights")
    assert variant.exists_locally() is True
