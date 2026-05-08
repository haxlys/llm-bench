"""Import ProgramBench eval JSON into llm-bench eval results.

This is intentionally eval-only: run ProgramBench's agent/submission workflow
elsewhere, then point this importer at the directory containing
``<instance_id>/<instance_id>.eval.json`` files.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from llm_bench.evals.programbench_runner import DEFAULT_ALMOST_RESOLVED_THRESHOLD
from llm_bench.evals.programbench_runner import TASK_NAME
from llm_bench.evals.programbench_runner import run_programbench_import
from llm_bench.registry import get_registry

ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = ROOT / "results" / "eval_scores"


def now_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@click.command()
@click.option("--variant", required=True, help="Registry variant key this ProgramBench run measured.")
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="ProgramBench output directory containing *.eval.json files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=EVAL_RESULTS_DIR,
    show_default=True,
    help="Root eval_scores directory to write into.",
)
@click.option("--limit", type=int, default=None, help="Import only the first N eval JSON files.")
@click.option(
    "--almost-threshold",
    type=float,
    default=DEFAULT_ALMOST_RESOLVED_THRESHOLD,
    show_default=True,
    help="Per-instance pass-rate threshold for almost_resolved.",
)
@click.option(
    "--tasks-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional ProgramBench data/tasks directory. When provided, ignored "
        "branches/tests from tests.json are excluded like `programbench info`."
    ),
)
def main(
    variant: str,
    source_dir: Path,
    output_dir: Path,
    limit: int | None,
    almost_threshold: float,
    tasks_dir: Path | None,
) -> None:
    try:
        get_registry().variant(variant)
    except KeyError as exc:
        raise click.BadParameter(str(exc), param_hint="--variant") from exc

    run_dir = output_dir / f"{now_safe()}_{variant}_full" / TASK_NAME
    result = run_programbench_import(
        source_dir=source_dir,
        output_dir=run_dir,
        limit=limit,
        almost_threshold=almost_threshold,
        tasks_dir=tasks_dir,
    )
    if "error" in result:
        click.echo(f"ProgramBench import failed: {result['error']}", err=True)
        sys.exit(1)

    metrics = result["results"][TASK_NAME]
    click.echo(f"→ {result['results_file']}")
    click.echo(
        "→ resolved="
        f"{metrics['resolved_rate,none']:.4f}, "
        "almost_resolved="
        f"{metrics['almost_resolved_rate,none']:.4f}, "
        "avg_test_pass_rate="
        f"{metrics['avg_test_pass_rate,none']:.4f}, "
        f"instances={metrics['n_instances,none']}"
    )


if __name__ == "__main__":
    main()
