"""MTPLX runner — measures native-MTP and target-only AR through `mtplx ask`."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from llm_bench import BENCH_VERSION
from llm_bench.runners.base import BenchResult, Scenario, now_iso, run_with_time


def _filler_prompt(n_prompt: int) -> str:
    """Approximate a prompt of n_prompt tokens without requiring a tokenizer."""
    base = "The quick brown fox jumps over the lazy dog. "
    word_target = max(1, int(n_prompt / 1.3))
    words_per_base = len(base.split())
    repeats = max(1, word_target // words_per_base + 1)
    return (base * repeats).strip()


def _mtplx_executable() -> str:
    venv_script = Path(sys.executable).with_name("mtplx")
    if venv_script.exists():
        return str(venv_script)
    return shutil.which("mtplx") or "mtplx"


def _json_object(stdout: str) -> dict:
    """Parse MTPLX JSON even if a wrapper emits incidental text."""
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stdout[start : end + 1])


def _acceptance_rates(accepted: list[int], drafted: list[int]) -> list[float]:
    rates: list[float] = []
    for a, d in zip(accepted, drafted):
        rates.append(round((a / d) if d else 0.0, 6))
    return rates


class MTPLXRunner:
    """Run the same model through MTPLX with native MTP on or target-only AR."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        quant: str = "MLX-4bit",
        variant_key: str = "",
        generation_mode: str = "mtp",
        profile: str = "performance-cold",
        depth: int = 3,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_fans: bool | None = None,
    ):
        if generation_mode not in {"mtp", "ar"}:
            raise ValueError("generation_mode must be 'mtp' or 'ar'")
        self.model_id = model_id
        self.model_path = model_path
        self.quant = quant
        self.variant_key = variant_key
        self.generation_mode = generation_mode
        self.profile = profile
        self.depth = depth
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_fans = _env_truthy("MTPLX_MAX") if max_fans is None else max_fans

    def run(self, scenario: Scenario, run_idx: int) -> BenchResult:
        prompt = _filler_prompt(scenario.n_prompt)
        cmd = [
            _mtplx_executable(),
            "ask",
            "--model", self.model_path,
            "--profile", self.profile,
            "--depth", str(self.depth),
            "--max-tokens", str(scenario.n_gen),
            "--temperature", str(self.temperature),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--reasoning", "off",
            "--json",
            "--yes",
            "--prompt", prompt,
            "--mtp" if self.generation_mode == "mtp" else "--no-mtp",
        ]
        if self.max_fans:
            cmd.append("--max")

        stdout, stderr, wall, peak_mem_gb = run_with_time(cmd, check=False)
        payload = _json_object(stdout)
        stats = payload.get("stats") if isinstance(payload, dict) else {}
        if not isinstance(stats, dict):
            stats = {}
        tok_s = float(stats.get("tok_s") or 0.0)
        if tok_s <= 0:
            raise RuntimeError(f"MTPLX JSON missing positive stats.tok_s: {stdout[-1500:]}")

        accepted = [int(x) for x in stats.get("accepted_by_depth", [])]
        drafted = [int(x) for x in stats.get("drafted_by_depth", [])]
        actual_mode = str(stats.get("generation_mode") or self.generation_mode)
        generated_tokens = int(stats.get("generated_tokens") or scenario.n_gen)

        return BenchResult(
            model_id=self.model_id,
            fmt="mlx",
            quant=self.quant,
            scenario=scenario.name,
            n_prompt=scenario.n_prompt,
            n_gen=generated_tokens,
            pp_tps=0.0,
            tg_tps=tok_s,
            peak_mem_gb=round(peak_mem_gb, 3),
            wall_s=round(wall, 3),
            run_idx=run_idx,
            ts=now_iso(),
            bench_version=BENCH_VERSION,
            variant_key=self.variant_key,
            backend="mtplx",
            artifact_type="hf_repo",
            generation_mode=actual_mode,
            raw={
                "measurement": "mtplx_ask_json",
                "generation_mode": actual_mode,
                "speedup_baseline": "target_only_ar",
                "profile": self.profile,
                "depth": self.depth,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_fans_requested": self.max_fans,
                "verify_ms_per_call": stats.get("verify_ms_per_call"),
                "verify_calls": stats.get("verify_calls"),
                "accepted_by_depth": accepted,
                "drafted_by_depth": drafted,
                "acceptance_rates_by_depth": _acceptance_rates(accepted, drafted),
                "stats": stats,
                "stderr_tail": stderr[-2000:],
            },
        )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
