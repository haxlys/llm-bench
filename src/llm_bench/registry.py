"""Model registry — declarative source of truth for what gets benchmarked.

Loads models/registry.yaml into typed dataclasses. All other code (runners,
aggregators, dashboard, sync_models) reads from this — no hardcoded variant
lists elsewhere.

Adding a new model variant: edit registry.yaml, no Python changes.
"""

from __future__ import annotations

import os
import warnings
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
    fmt: str                      # legacy artifact/runtime label ("mlx", "gguf", "api", ...)
    path: str                     # HF repo id (mlx) OR local file path (gguf)
    quant: str                    # "MLX-8bit", "Q8_0", "Q4_K_M", ...
    tier: str                     # "8bit", "4bit" — used to pair MLX↔GGUF
    backend: str = ""             # runtime adapter key; defaults to fmt
    artifact_type: str = ""       # e.g. hf_repo, gguf_file, endpoint
    capabilities: frozenset[str] = field(default_factory=frozenset)
    api_model: str = ""           # model id sent to OpenAI-compatible APIs
    tokenizer: str = ""           # HF tokenizer repo/path for completion evals
    api_key_env: str = ""         # env var copied to Authorization/OpenAI_API_KEY
    generation_mode: str = ""     # "mtp" / "ar" for MTPLX comparison variants
    params_total_b: float | None = None
    params_active_b: float | None = None
    approx_size_gb: float | None = None
    download: Download | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.backend:
            object.__setattr__(self, "backend", self.fmt)
        if not self.artifact_type:
            object.__setattr__(self, "artifact_type", default_artifact_type(self.fmt))
        if not self.capabilities:
            object.__setattr__(
                self,
                "capabilities",
                default_capabilities(self.backend, self.fmt),
            )

    @property
    def is_local_file(self) -> bool:
        """True if path points to an on-disk model artifact."""
        return self.artifact_type in {"gguf_file", "file"}

    @property
    def requires_local_artifact(self) -> bool:
        """False for hosted endpoints that have no local download/check step."""
        return self.artifact_type != "endpoint"

    @property
    def api_model_label(self) -> str:
        """Model label to send to OpenAI-compatible eval clients."""
        if self.api_model:
            return self.api_model
        if self.backend == "openai-compatible":
            return self.model_id
        if self.backend == "mlx":
            return self.path
        return self.key

    @property
    def resolved_path(self) -> str:
        """Expand local paths; keep repo ids/endpoints as-is."""
        if self.is_local_file:
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
        if not self.requires_local_artifact:
            return True
        if self.is_local_file:
            p = Path(self.resolved_path)
            return p.is_file() and p.stat().st_size > 0
        if self.artifact_type != "hf_repo":
            return False
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
        if not config.is_file() or config.stat().st_size <= 0:
            return False
        snapshot = config.parent
        weight_globs = ("*.safetensors", "*.bin", "*.gguf")
        return any(
            p.is_file() and p.stat().st_size > 0
            for pattern in weight_globs
            for p in snapshot.glob(pattern)
        )


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

    def variants_by_backend(self, backend: str) -> list[Variant]:
        return [v for v in self.variants if v.backend == backend]

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


ALLOWED_FMTS = {"mlx", "gguf", "api"}
ALLOWED_TIERS = {"4bit", "5bit", "6bit", "8bit", "16bit", "fp16", "bf16", "hosted"}
ALLOWED_ARCHS = {"dense", "moe"}


def default_artifact_type(fmt: str) -> str:
    if fmt == "gguf":
        return "gguf_file"
    if fmt == "mlx":
        return "hf_repo"
    if fmt == "api":
        return "endpoint"
    return "artifact"


def default_capabilities(backend: str, fmt: str) -> frozenset[str]:
    key = backend or fmt
    if key == "gguf":
        return frozenset({"chat", "completions", "code_eval_chat", "tool_use_eval"})
    if key == "mlx":
        return frozenset({"chat", "completions"})
    if key == "openai-compatible":
        return frozenset({"chat", "completions"})
    if key == "mtplx":
        return frozenset({"chat", "completions"})
    return frozenset()


def _validate(variants: list[Variant], models: list["Model"]) -> None:
    seen: dict[str, str] = {}
    unpinned: list[str] = []
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
        if v.backend == "mtplx" and v.generation_mode not in {"mtp", "ar"}:
            raise ValueError(
                f"variant '{v.key}': mtplx backend requires "
                "generation_mode='mtp' or 'ar'"
            )
        # Local files without download spec must already have an absolute path on disk.
        if v.is_local_file and v.download is None:
            if not v.resolved_path.startswith("/"):
                raise ValueError(
                    f"variant '{v.key}': local artifact path '{v.path}' is relative "
                    f"and has no download: spec; provide one or use absolute path"
                )
        # Reproducibility: warn (don't fail) when a remote source has no pinned
        # revision. A reproducer two months out may pick up new weights at the
        # same key — same BENCH_VERSION, different numbers.
        needs_pin = v.artifact_type == "hf_repo" or (
            v.is_local_file and v.download is not None
        )
        has_pin = v.download is not None and v.download.revision is not None
        if needs_pin and not has_pin:
            unpinned.append(v.key)
    if unpinned:
        warnings.warn(
            f"{len(unpinned)} variant(s) have no pinned revision; weights may "
            f"change under your feet across reproducer runs: "
            f"{sorted(unpinned)}. Add `revision: <commit-sha>` under download.",
            UserWarning,
            stacklevel=3,
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
            fmt = v_raw["fmt"]
            backend = v_raw.get("backend") or fmt
            artifact_type = v_raw.get("artifact_type") or default_artifact_type(fmt)
            caps_raw = v_raw.get("capabilities")
            capabilities = (
                frozenset(str(c) for c in caps_raw)
                if caps_raw is not None
                else default_capabilities(backend, fmt)
            )
            v_objs.append(Variant(
                key=v_raw["key"],
                model_id=m_raw["id"],
                family=m_raw["family"],
                architecture=m_raw["architecture"],
                fmt=fmt,
                path=_interpolate(v_raw["path"], defaults),
                quant=v_raw["quant"],
                tier=v_raw["tier"],
                backend=backend,
                artifact_type=artifact_type,
                capabilities=capabilities,
                api_model=v_raw.get("api_model", ""),
                tokenizer=v_raw.get("tokenizer", ""),
                api_key_env=v_raw.get("api_key_env", ""),
                generation_mode=v_raw.get("generation_mode", ""),
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
