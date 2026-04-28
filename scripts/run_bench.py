"""Run the MLX vs GGUF benchmark matrix.

Example:
    uv run python scripts/run_bench.py \
        --mlx-model lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit \
        --gguf-model ~/models/gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
        --model-id gemma-4-26B-A4B-it
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# allow `uv run python scripts/run_bench.py` without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_bench.aggregate import write_summary  # noqa: E402
from llm_bench.runners import GGUFRunner, MLXRunner  # noqa: E402
from llm_bench.runners.base import write_raw  # noqa: E402
from llm_bench.scenarios import default_scenarios, smoke_scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_CSV = ROOT / "results" / "summary.csv"


@click.command()
@click.option("--mlx-model", default="lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit",
              help="MLX model: HF repo id or local path")
@click.option("--gguf-model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to .gguf file")
@click.option("--model-id", default="gemma-4-26B-A4B-it",
              help="Logical model id used for grouping")
@click.option("--mlx-quant", default="MLX-8bit")
@click.option("--gguf-quant", default="Q8_0")
@click.option("--runs", default=3, show_default=True, help="Measured runs per scenario")
@click.option("--warmup/--no-warmup", default=True,
              help="Run one extra warmup run (not recorded)")
@click.option("--smoke/--full", default=False, help="Smoke = single scenario only")
@click.option("--only", type=click.Choice(["mlx", "gguf", "both"]), default="both")
def main(
    mlx_model: str,
    gguf_model: str,
    model_id: str,
    mlx_quant: str,
    gguf_quant: str,
    runs: int,
    warmup: bool,
    smoke: bool,
    only: str,
):
    scenarios = smoke_scenarios() if smoke else default_scenarios()
    runners = []
    if only in ("mlx", "both"):
        runners.append(MLXRunner(model_id=model_id, model_path=mlx_model, quant=mlx_quant))
    if only in ("gguf", "both"):
        runners.append(GGUFRunner(model_id=model_id, model_path=gguf_model, quant=gguf_quant))

    total = len(runners) * len(scenarios) * (runs + (1 if warmup else 0))
    click.echo(f"→ {total} invocations ({len(runners)} runners × {len(scenarios)} scenarios × "
               f"{runs} runs{' + warmup' if warmup else ''})")

    for runner in runners:
        for sc in scenarios:
            if warmup:
                click.echo(f"  [warmup] {runner.__class__.__name__} {sc.name}")
                try:
                    runner.run(sc, run_idx=0)
                except Exception as e:
                    click.echo(f"    ! warmup failed: {e}", err=True)
                    continue
            for i in range(1, runs + 1):
                click.echo(f"  [run {i}] {runner.__class__.__name__} {sc.name} ", nl=False)
                try:
                    res = runner.run(sc, run_idx=i)
                    write_raw(res, RAW_DIR)
                    click.echo(f"pp={res.pp_tps:.1f} tg={res.tg_tps:.1f} mem={res.peak_mem_gb:.1f}GB")
                except Exception as e:
                    click.echo(f"FAILED: {e}", err=True)

    out = write_summary(RAW_DIR, SUMMARY_CSV)
    click.echo(f"\n→ summary written to {out}")


if __name__ == "__main__":
    main()
