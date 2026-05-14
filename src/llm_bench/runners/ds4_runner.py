"""DS4 runner — invokes the DeepSeek V4 Flash-specific `ds4-bench`."""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

from llm_bench import BENCH_VERSION
from llm_bench.runners.base import BenchResult, Scenario, now_iso, run_with_time


class DS4Runner:
    def __init__(
        self,
        model_id: str,
        model_path: str,
        quant: str = "IQ2XXS",
        n_threads: int | None = None,
        variant_key: str = "",
        backend_name: str = "metal",
        prompt_file: str | None = None,
        runtime_root: str = "",
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.quant = quant
        self.n_threads = n_threads
        self.variant_key = variant_key
        self.backend_name = backend_name
        self.ds4_root = _resolve_ds4_root(model_path, runtime_root)
        self.ds4_bench = _find_ds4_binary("ds4-bench", self.ds4_root)
        self.prompt_file = (
            Path(prompt_file).expanduser()
            if prompt_file
            else self.ds4_root / "speed-bench" / "promessi_sposi.txt"
        )
        if not self.prompt_file.is_file():
            raise RuntimeError(f"DS4 benchmark prompt not found: {self.prompt_file}")

    def run(self, scenario: Scenario, run_idx: int) -> BenchResult:
        ctx_alloc = scenario.n_prompt + scenario.n_gen + 1
        cmd = [
            self.ds4_bench,
            "-m", self.model_path,
            "--prompt-file", str(self.prompt_file),
            "--backend", self.backend_name,
            "--ctx-start", str(scenario.n_prompt),
            "--ctx-max", str(scenario.n_prompt),
            "--ctx-alloc", str(ctx_alloc),
            "--step-incr", str(max(scenario.n_prompt, 1)),
            "--gen-tokens", str(scenario.n_gen),
        ]
        if self.n_threads:
            cmd.extend(["-t", str(self.n_threads)])

        stdout, stderr, wall, peak_mem_gb = run_with_time(cmd, cwd=str(self.ds4_root))
        row = _parse_single_row(stdout)
        pp_tps = float(row["prefill_tps"])
        tg_tps = float(row["gen_tps"])

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
            bench_version=BENCH_VERSION,
            variant_key=self.variant_key,
            backend="ds4",
            artifact_type="gguf_file",
            generation_mode="ar",
            raw={
                "ds4_bench_row": row,
                "prompt_file": str(self.prompt_file),
                "stderr_tail": stderr[-2000:],
            },
        )


def _find_ds4_binary(name: str, ds4_root: Path) -> str:
    local = ds4_root / name
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} not found. Build ds4 or put {name} on PATH.")


def _resolve_ds4_root(model_path: str, runtime_root: str = "") -> Path:
    if runtime_root:
        return Path(runtime_root).expanduser().resolve()
    return Path(model_path).expanduser().resolve().parent.parent


def _parse_single_row(stdout: str) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one ds4-bench CSV row, got {len(rows)}:\n{stdout[-1500:]}")
    row = rows[0]
    required = {"prefill_tps", "gen_tps", "kvcache_bytes"}
    missing = required - set(row)
    if missing:
        raise RuntimeError(f"Missing ds4-bench column(s) {sorted(missing)}:\n{stdout[-1500:]}")
    return row
