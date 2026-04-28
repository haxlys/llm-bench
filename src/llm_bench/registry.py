"""Model registry — declarative source of truth for what gets benchmarked.

Loads models/registry.yaml into typed dataclasses. All other code (runners,
aggregators, dashboard, sync_models) reads from this — no hardcoded variant
lists elsewhere.

Adding a new model variant: edit registry.yaml, no Python changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "registry.yaml"


@dataclass(frozen=True)
class Download:
    """How to fetch a variant from Hugging Face."""
    repo: str                # e.g. "unsloth/gemma-4-26B-A4B-it-GGUF"
    pattern: str | None = None    # glob pattern for --include filter
    revision: str | None = None   # optional pinned revision


@dataclass(frozen=True)
class Variant:
    key: str                      # unique identifier ("26B-MoE-mlx-8bit")
    model_id: str                 # logical model id ("gemma-4-26B-A4B-it")
    family: str                   # "gemma", "qwen", "llama"
    architecture: Literal["dense", "moe"]
    fmt: Literal["mlx", "gguf"]
    path: str                     # HF repo id (mlx) OR local file path (gguf)
    quant: str                    # "MLX-8bit", "Q8_0", "Q4_K_M", ...
    tier: str                     # "8bit", "4bit" — used to pair MLX↔GGUF
    params_total_b: float | None = None
    params_active_b: float | None = None
    approx_size_gb: float | None = None
    download: Download | None = None
    notes: str = ""

    @property
    def is_local_file(self) -> bool:
        """True if path points to an on-disk file (gguf), False if HF repo id (mlx)."""
        return self.fmt == "gguf"

    @property
    def resolved_path(self) -> str:
        """For gguf: expanded absolute path. For mlx: HF id passthrough."""
        if self.fmt == "gguf":
            return os.path.expanduser(self.path)
        return self.path

    def exists_locally(self) -> bool:
        """Verify the local artifact is present AND complete enough to load.

        For gguf: file on disk.
        For mlx: HF cache has refs/main → snapshots/<rev>/config.json (signal
            that at least the metadata reached disk; partial weight downloads
            are still risky but config.json present means the snapshot dir
            was created after the resolve step).
        """
        if self.fmt == "gguf":
            p = Path(self.resolved_path)
            return p.is_file() and p.stat().st_size > 0
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + self.path.replace("/", "--")
        repo_dir = cache_dir / slug
        refs_main = repo_dir / "refs" / "main"
        if not refs_main.is_file():
            return False
        try:
            rev = refs_main.read_text().strip()
        except OSError:
            return False
        config = repo_dir / "snapshots" / rev / "config.json"
        return config.is_file() and config.stat().st_size > 0


@dataclass(frozen=True)
class Model:
    id: str
    family: str
    architecture: Literal["dense", "moe"]
    variants: tuple[Variant, ...] = field(default_factory=tuple)
    params_total_b: float | None = None
    params_active_b: float | None = None
    notes: str = ""


@dataclass
class Registry:
    models: tuple[Model, ...]
    defaults: dict

    # --- lookups ---
    @property
    def variants(self) -> list[Variant]:
        return [v for m in self.models for v in m.variants]

    def variant(self, key: str) -> Variant:
        for v in self.variants:
            if v.key == key:
                return v
        raise KeyError(f"variant '{key}' not in registry. "
                       f"Known keys: {[v.key for v in self.variants]}")

    def variants_by_model(self, model_id: str) -> list[Variant]:
        return [v for v in self.variants if v.model_id == model_id]

    def variants_by_tier(self, tier: str) -> list[Variant]:
        return [v for v in self.variants if v.tier == tier]

    def variants_by_fmt(self, fmt: str) -> list[Variant]:
        return [v for v in self.variants if v.fmt == fmt]

    def model(self, model_id: str) -> Model:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"model '{model_id}' not in registry")

    def variant_keys(self) -> list[str]:
        return [v.key for v in self.variants]


def _interpolate(s: str, defaults: dict) -> str:
    """Replace {var} occurrences using defaults. Idempotent for plain strings."""
    if not isinstance(s, str):
        return s
    out = s
    for k, v in defaults.items():
        out = out.replace("{" + k + "}", str(v))
    return os.path.expanduser(out)


ALLOWED_FMTS = {"mlx", "gguf"}
ALLOWED_TIERS = {"4bit", "5bit", "6bit", "8bit", "16bit", "fp16", "bf16"}
ALLOWED_ARCHS = {"dense", "moe"}


def _validate(variants: list[Variant], models: list["Model"]) -> None:
    seen: dict[str, str] = {}
    for v in variants:
        if v.key in seen:
            raise ValueError(
                f"duplicate variant key '{v.key}' "
                f"(used by both '{seen[v.key]}' and '{v.model_id}')"
            )
        seen[v.key] = v.model_id
        if v.fmt not in ALLOWED_FMTS:
            raise ValueError(
                f"variant '{v.key}': fmt='{v.fmt}' not in {sorted(ALLOWED_FMTS)}"
            )
        if v.tier not in ALLOWED_TIERS:
            raise ValueError(
                f"variant '{v.key}': tier='{v.tier}' not in {sorted(ALLOWED_TIERS)}"
            )
        if v.architecture not in ALLOWED_ARCHS:
            raise ValueError(
                f"variant '{v.key}': architecture='{v.architecture}' "
                f"not in {sorted(ALLOWED_ARCHS)}"
            )
        # GGUF without download spec must already have an absolute path on disk
        if v.fmt == "gguf" and v.download is None:
            if not v.resolved_path.startswith("/"):
                raise ValueError(
                    f"variant '{v.key}': gguf path '{v.path}' is relative "
                    f"and has no download: spec; provide one or use absolute path"
                )


def load_registry(path: Path | None = None) -> Registry:
    """Parse registry.yaml into typed Models/Variants. Validates uniqueness."""
    yaml_path = path or REGISTRY_PATH
    raw = yaml.safe_load(yaml_path.read_text())
    defaults = raw.get("defaults", {}) or {}

    models: list[Model] = []
    for m_raw in raw.get("models", []):
        v_objs: list[Variant] = []
        for v_raw in m_raw.get("variants", []):
            dl_raw = v_raw.get("download")
            dl = Download(**dl_raw) if dl_raw else None
            v_objs.append(Variant(
                key=v_raw["key"],
                model_id=m_raw["id"],
                family=m_raw["family"],
                architecture=m_raw["architecture"],
                fmt=v_raw["fmt"],
                path=_interpolate(v_raw["path"], defaults),
                quant=v_raw["quant"],
                tier=v_raw["tier"],
                params_total_b=m_raw.get("params_total_b") or v_raw.get("params_total_b"),
                params_active_b=m_raw.get("params_active_b") or v_raw.get("params_active_b"),
                approx_size_gb=v_raw.get("approx_size_gb"),
                download=dl,
                notes=v_raw.get("notes", m_raw.get("notes", "")),
            ))
        models.append(Model(
            id=m_raw["id"],
            family=m_raw["family"],
            architecture=m_raw["architecture"],
            params_total_b=m_raw.get("params_total_b"),
            params_active_b=m_raw.get("params_active_b"),
            notes=m_raw.get("notes", ""),
            variants=tuple(v_objs),
        ))

    all_variants = [v for m in models for v in m.variants]
    _validate(all_variants, models)
    return Registry(models=tuple(models), defaults=defaults)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """Cached registry singleton."""
    return load_registry()
