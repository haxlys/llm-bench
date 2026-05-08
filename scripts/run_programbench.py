"""Run ProgramBench evaluation and import the results into llm-bench.

This script expects a ProgramBench submission run directory shaped like:

    <run-dir>/<instance_id>/submission.tar.gz

It delegates Docker evaluation to the upstream `programbench eval` CLI, then
converts the emitted `*.eval.json` files into llm-bench's synthetic eval result
format so the normal aggregation and site-data pipeline can consume them.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
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
PROGRAMBENCH_RUNS_DIR = ROOT / "results" / "programbench_runs"


def now_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _evaluated_source_dir(source_dir: Path, programbench_output_dir: Path) -> Path:
    return programbench_output_dir / source_dir.name


def _programbench_env(blob_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if blob_dir is not None:
        env["PROGRAMBENCH_BLOB_DIR"] = str(blob_dir)
    return env


def _programbench_eval_command(
    *,
    programbench_bin: str,
    source_dir: Path,
    programbench_output_dir: Path,
    workers: int,
    branch_workers: int,
    docker_cpus: int,
    branch_retries: int,
    force: bool,
    filter_spec: str,
    slice_spec: str,
    image_tag: str,
) -> list[str]:
    cmd = shlex.split(programbench_bin) + [
        "eval",
        str(source_dir),
        "--workers",
        str(workers),
        "--branch-workers",
        str(branch_workers),
        "--docker-cpus",
        str(docker_cpus),
        "--branch-retries",
        str(branch_retries),
        "--image-tag",
        image_tag,
        "--output",
        str(programbench_output_dir),
    ]
    if force:
        cmd.append("--force")
    if filter_spec:
        cmd.extend(["--filter", filter_spec])
    if slice_spec:
        cmd.extend(["--slice", slice_spec])
    return cmd


def _ensure_programbench_command(cmd: list[str]) -> None:
    if not cmd:
        raise click.ClickException("empty ProgramBench command")
    if shutil.which(cmd[0]) is None:
        raise click.ClickException(
            "ProgramBench CLI was not found. Install it with "
            "`uv sync --extra programbench`, `uv pip install programbench`, "
            "or pass --programbench-bin."
        )


@click.command()
@click.option("--variant", required=True, help="Registry variant key this ProgramBench run measured.")
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing <instance_id>/submission.tar.gz ProgramBench submissions.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=EVAL_RESULTS_DIR,
    show_default=True,
    help="Root eval_scores directory to write imported llm-bench results into.",
)
@click.option(
    "--programbench-output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Root directory for raw ProgramBench eval JSON. Defaults to a timestamped results/programbench_runs directory.",
)
@click.option(
    "--programbench-bin",
    default="programbench",
    show_default=True,
    help="ProgramBench executable command. Quoted commands like 'uvx programbench' are accepted.",
)
@click.option("--workers", type=int, default=1, show_default=True, help="ProgramBench instance workers.")
@click.option(
    "--branch-workers",
    type=int,
    default=1,
    show_default=True,
    help="ProgramBench branch workers per instance.",
)
@click.option(
    "--docker-cpus",
    type=int,
    default=4,
    show_default=True,
    help="CPU cores per ProgramBench Docker container.",
)
@click.option(
    "--branch-retries",
    type=int,
    default=1,
    show_default=True,
    help="Retries for ProgramBench test branches with worker crashes.",
)
@click.option("--force", is_flag=True, help="Re-evaluate even if ProgramBench eval JSON exists.")
@click.option("--filter", "filter_spec", default="", help="ProgramBench instance-id regex filter.")
@click.option("--slice", "slice_spec", default="", help="ProgramBench slice spec, e.g. 0:5.")
@click.option("--image-tag", default="task", show_default=True, help="ProgramBench Docker image tag.")
@click.option("--limit", type=int, default=None, help="Import only the first N eval JSON files after eval.")
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
@click.option(
    "--blob-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional PROGRAMBENCH_BLOB_DIR override for pre-downloaded test blobs.",
)
def main(
    variant: str,
    source_dir: Path,
    output_dir: Path,
    programbench_output_dir: Path | None,
    programbench_bin: str,
    workers: int,
    branch_workers: int,
    docker_cpus: int,
    branch_retries: int,
    force: bool,
    filter_spec: str,
    slice_spec: str,
    image_tag: str,
    limit: int | None,
    almost_threshold: float,
    tasks_dir: Path | None,
    blob_dir: Path | None,
) -> None:
    try:
        get_registry().variant(variant)
    except KeyError as exc:
        raise click.BadParameter(str(exc), param_hint="--variant") from exc

    run_id = f"{now_safe()}_{variant}_programbench"
    raw_output_dir = programbench_output_dir or PROGRAMBENCH_RUNS_DIR / run_id
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = _programbench_eval_command(
        programbench_bin=programbench_bin,
        source_dir=source_dir,
        programbench_output_dir=raw_output_dir,
        workers=workers,
        branch_workers=branch_workers,
        docker_cpus=docker_cpus,
        branch_retries=branch_retries,
        force=force,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        image_tag=image_tag,
    )
    _ensure_programbench_command(cmd)

    click.echo("Running ProgramBench:")
    click.echo(f"  {shlex.join(cmd)}")
    completed = subprocess.run(cmd, check=False, env=_programbench_env(blob_dir))
    if completed.returncode != 0:
        click.echo(f"ProgramBench failed with exit code {completed.returncode}", err=True)
        sys.exit(completed.returncode)

    evaluated_dir = _evaluated_source_dir(source_dir, raw_output_dir)
    run_dir = output_dir / f"{run_id}_full" / TASK_NAME
    result = run_programbench_import(
        source_dir=evaluated_dir,
        output_dir=run_dir,
        limit=limit,
        almost_threshold=almost_threshold,
        tasks_dir=tasks_dir,
    )
    if "error" in result:
        click.echo(f"ProgramBench import failed: {result['error']}", err=True)
        sys.exit(1)

    metrics = result["results"][TASK_NAME]
    click.echo(f"Raw ProgramBench output: {evaluated_dir}")
    click.echo(f"Imported llm-bench result: {result['results_file']}")
    click.echo(
        "Metrics: "
        f"resolved={metrics['resolved_rate,none']:.4f}, "
        f"almost_resolved={metrics['almost_resolved_rate,none']:.4f}, "
        f"avg_test_pass_rate={metrics['avg_test_pass_rate,none']:.4f}, "
        f"instances={metrics['n_instances,none']}"
    )


if __name__ == "__main__":
    main()
