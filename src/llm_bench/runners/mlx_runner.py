"""MLX runner — invokes `mlx_lm.generate` CLI as subprocess, parses verbose output."""

from __future__ import annotations

import re
import sys

from llm_bench import BENCH_VERSION
from llm_bench.runners.base import BenchResult, Scenario, now_iso, run_with_time

# Sample mlx_lm verbose output lines we parse:
#   Prompt: 256 tokens, 412.345 tokens-per-sec
#   Generation: 128 tokens, 53.21 tokens-per-sec
#   Peak memory: 28.412 GB
_PP_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec")
_TG_RE = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec")
_MEM_RE = re.compile(r"Peak memory:\s+([\d.]+)\s+GB")


def _filler_prompt(n_prompt: int) -> str:
    """Approx n_prompt tokens of filler. mlx tokenizer averages ~1.3 tok/word for English.

    We aim slightly under and let actual tokenization decide; the runner records the
    actual prompt token count from mlx_lm's verbose output.
    """
    base = "The quick brown fox jumps over the lazy dog. "
    word_target = max(1, int(n_prompt / 1.3))
    words_per_base = len(base.split())
    repeats = max(1, word_target // words_per_base + 1)
    return (base * repeats).strip()


class MLXRunner:
    def __init__(self, model_id: str, model_path: str, quant: str = "8bit",
                 variant_key: str = ""):
        self.model_id = model_id
        self.model_path = model_path  # HF repo id or local path
        self.quant = quant
        self.variant_key = variant_key

    def run(self, scenario: Scenario, run_idx: int) -> BenchResult:
        prompt = _filler_prompt(scenario.n_prompt)
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", self.model_path,
            "--prompt", prompt,
            "--max-tokens", str(scenario.n_gen),
            "--temp", "0.0",
            "--verbose", "True",
        ]
        stdout, stderr, wall, peak_mem_gb = run_with_time(cmd)
        combined = stdout + "\n" + stderr

        pp_match = _PP_RE.search(combined)
        tg_match = _TG_RE.search(combined)
        mem_match = _MEM_RE.search(combined)
        if not (pp_match and tg_match):
            raise RuntimeError(
                f"Failed to parse mlx_lm output for scenario={scenario.name}.\n"
                f"stdout tail:\n{stdout[-1500:]}\nstderr tail:\n{stderr[-1500:]}"
            )

        actual_n_prompt = int(pp_match.group(1))
        pp_tps = float(pp_match.group(2))
        actual_n_gen = int(tg_match.group(1))
        tg_tps = float(tg_match.group(2))
        # Prefer mlx's own peak mem (Metal-aware); fall back to time -l RSS.
        mlx_peak = float(mem_match.group(1)) if mem_match else 0.0
        peak = max(mlx_peak, peak_mem_gb)

        return BenchResult(
            model_id=self.model_id,
            fmt="mlx",
            quant=self.quant,
            scenario=scenario.name,
            n_prompt=actual_n_prompt,
            n_gen=actual_n_gen,
            pp_tps=pp_tps,
            tg_tps=tg_tps,
            peak_mem_gb=round(peak, 3),
            wall_s=round(wall, 3),
            run_idx=run_idx,
            ts=now_iso(),
            bench_version=BENCH_VERSION,
            variant_key=self.variant_key,
            backend="mlx",
            artifact_type="hf_repo",
            raw={"mlx_peak_gb": mlx_peak, "rss_peak_gb": peak_mem_gb},
        )
