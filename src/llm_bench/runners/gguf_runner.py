"""GGUF runner — invokes `llama-bench` for tok/s and `llama-cli` for memory."""

from __future__ import annotations

import json
import shutil

from llm_bench.runners.base import BenchResult, Scenario, now_iso, run_with_time


class GGUFRunner:
    def __init__(
        self,
        model_id: str,
        model_path: str,
        quant: str = "Q8_0",
        n_threads: int | None = None,
        n_gpu_layers: int = 999,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.quant = quant
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        if not shutil.which("llama-bench"):
            raise RuntimeError("llama-bench not found on PATH. brew install llama.cpp")

    def run(self, scenario: Scenario, run_idx: int) -> BenchResult:
        # llama-bench: synthetic prefill (-p) and generation (-n) measured separately,
        # then we take pp from the -p run and tg from the -n run in one invocation.
        cmd = [
            "llama-bench",
            "-m", self.model_path,
            "-p", str(scenario.n_prompt),
            "-n", str(scenario.n_gen),
            "-r", "1",
            "-ngl", str(self.n_gpu_layers),
            "-o", "json",
        ]
        if self.n_threads:
            cmd.extend(["-t", str(self.n_threads)])

        stdout, stderr, wall, peak_mem_gb = run_with_time(cmd)
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"llama-bench JSON parse failed: {e}\nstdout tail:\n{stdout[-1500:]}"
            ) from e

        pp_tps = 0.0
        tg_tps = 0.0
        for entry in data:
            n_p = entry.get("n_prompt", 0)
            n_g = entry.get("n_gen", 0)
            avg_ts = float(entry.get("avg_ts", 0.0))
            if n_p > 0 and n_g == 0:
                pp_tps = avg_ts
            elif n_g > 0 and n_p == 0:
                tg_tps = avg_ts

        if pp_tps == 0.0 or tg_tps == 0.0:
            raise RuntimeError(f"Missing pp/tg in llama-bench output: {data}")

        return BenchResult(
            model_id=self.model_id,
            fmt="gguf",
            quant=self.quant,
            scenario=scenario.name,
            n_prompt=scenario.n_prompt,
            n_gen=scenario.n_gen,
            pp_tps=pp_tps,
            tg_tps=tg_tps,
            peak_mem_gb=round(peak_mem_gb, 3),
            wall_s=round(wall, 3),
            run_idx=run_idx,
            ts=now_iso(),
            raw={"llama_bench_entries": data},
        )
