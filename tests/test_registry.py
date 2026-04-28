"""Tests for models/registry.yaml loading + validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.registry import Registry, Variant, get_registry, load_registry


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
