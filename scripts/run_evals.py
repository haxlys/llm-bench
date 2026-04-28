"""Run multi-dimensional evals for a model variant.

Boots a per-model OpenAI-compatible server, runs the requested suite via
lm-eval-harness, then tears down. Repeats per (model_id, fmt, quant).

Examples:
    # Smoke (limit=3 per task) on one model — verifies wiring
    uv run python scripts/run_evals.py --smoke \
        --variant gemma-4-26B-A4B-it:mlx:lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit

    # Full overnight run on all six variants
    uv run python scripts/run_evals.py --suite full --all-variants
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench.evals import ModelServer, full_suite, smoke_suite  # noqa: E402
from llm_bench.evals.bfcl import run_bfcl  # noqa: E402
from llm_bench.evals.lmeval import run_lmeval  # noqa: E402

EVAL_RESULTS_DIR = ROOT / "results" / "eval_scores"
SERVER_LOG_DIR = ROOT / "results" / "server_logs"


# (logical_id, fmt, model_path, quant_label, port)
VARIANTS = {
    "26B-MoE-mlx-8bit":  ("gemma-4-26B-A4B-it", "mlx", "lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit", "MLX-8bit"),
    "26B-MoE-mlx-4bit":  ("gemma-4-26B-A4B-it", "mlx", "lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit", "MLX-4bit"),
    "26B-MoE-gguf-q8":   ("gemma-4-26B-A4B-it", "gguf", str(Path.home() / "models/gguf/gemma-4-26B-A4B-it-Q8_0.gguf"), "Q8_0"),
    "26B-MoE-gguf-q4":   ("gemma-4-26B-A4B-it", "gguf", str(Path.home() / "models/gguf/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"), "Q4_K_M"),
    "31B-Dense-mlx-8bit":("gemma-4-31B-it",     "mlx", "lmstudio-community/gemma-4-31B-it-MLX-8bit", "MLX-8bit"),
    "31B-Dense-gguf-q8": ("gemma-4-31B-it",     "gguf", str(Path.home() / "models/gguf/gemma-4-31B-it-Q8_0.gguf"), "Q8_0"),
}


def now_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@click.command()
@click.option("--variant", multiple=True, type=click.Choice(list(VARIANTS.keys())),
              help="Variant key (repeatable). Default: just 26B-MoE-mlx-8bit for smoke.")
@click.option("--all-variants", is_flag=True, help="Run every variant in VARIANTS.")
@click.option("--suite", type=click.Choice(["smoke", "full"]), default="smoke",
              show_default=True)
@click.option("--limit", type=int, default=None,
              help="Override per-task sample limit. Smoke default = 3.")
@click.option("--port", type=int, default=9090, show_default=True)
@click.option("--include-bfcl", is_flag=True,
              help="Include BFCL (requires --bfcl-dir).")
@click.option("--bfcl-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
def main(variant: tuple, all_variants: bool, suite: str, limit: int | None,
         port: int, include_bfcl: bool, bfcl_dir: Path | None):
    if all_variants:
        keys = list(VARIANTS.keys())
    elif variant:
        keys = list(variant)
    else:
        keys = ["26B-MoE-mlx-8bit"]
        click.echo("→ no --variant given, defaulting to 26B-MoE-mlx-8bit")

    tasks = smoke_suite() if suite == "smoke" else full_suite()
    effective_limit = limit if limit is not None else (3 if suite == "smoke" else None)

    click.echo(f"→ {len(keys)} variants × {len(tasks)} tasks "
               f"(suite={suite}, limit={effective_limit})")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    grand_summary: list[dict] = []
    for key in keys:
        model_id, fmt, model_path, quant = VARIANTS[key]
        run_id = f"{now_safe()}_{key}_{suite}"
        out_dir = EVAL_RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        server_log = SERVER_LOG_DIR / f"{run_id}.log"

        click.echo(f"\n=== {key} ({fmt}, {quant}) ===")
        t0 = time.perf_counter()
        try:
            with ModelServer(fmt=fmt, model_path=model_path, port=port,
                             log_file=server_log) as base_url:
                click.echo(f"  server: {base_url} (booted in {time.perf_counter()-t0:.1f}s)")
                for dim, task in tasks:
                    click.echo(f"  [{dim}] {task} ", nl=False)
                    ts0 = time.perf_counter()
                    res = run_lmeval(
                        task=task, base_url=base_url, model_label=key,
                        output_dir=out_dir / task, limit=effective_limit,
                    )
                    dt = time.perf_counter() - ts0
                    if "error" in res:
                        click.echo(f"FAIL ({dt:.0f}s): {res['error']}")
                    else:
                        click.echo(f"OK ({dt:.0f}s)")
                    grand_summary.append({
                        "variant": key, "model_id": model_id, "fmt": fmt, "quant": quant,
                        "dim": dim, "task": task, "wall_s": round(dt, 1), **res,
                    })
                if include_bfcl:
                    click.echo(f"  [tool] bfcl ", nl=False)
                    res = run_bfcl(base_url, key, out_dir / "bfcl", bfcl_dir, effective_limit)
                    click.echo(res.get("status", res.get("error", "OK")))
                    grand_summary.append({"variant": key, "dim": "tool", "task": "bfcl", **res})
        except Exception as e:
            click.echo(f"  ! variant {key} aborted: {e}", err=True)
            grand_summary.append({"variant": key, "fatal": str(e)})

    summary_path = EVAL_RESULTS_DIR / f"summary_{now_safe()}_{suite}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(grand_summary, indent=2, ensure_ascii=False))
    click.echo(f"\n→ summary: {summary_path}")


if __name__ == "__main__":
    main()
