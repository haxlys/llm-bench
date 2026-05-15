"""Run Terminal-Bench against one llm-bench registry variant.

Terminal-Bench is Docker-backed and agentic, so this script defaults to one
task unless the caller explicitly asks for more. Results are converted into the
same synthetic ``results_*.json`` shape as the other eval runners.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from llm_bench.evals import ModelServer
from llm_bench.evals.terminal_bench_runner import DEFAULT_AGENT
from llm_bench.evals.terminal_bench_runner import DEFAULT_BIN
from llm_bench.evals.terminal_bench_runner import DEFAULT_DATASET
from llm_bench.evals.terminal_bench_runner import TASK_NAME
from llm_bench.evals.terminal_bench_runner import run_terminal_bench
from llm_bench.registry import get_registry, is_speed_only_variant

ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = ROOT / "results" / "eval_scores"
SERVER_LOG_DIR = ROOT / "results" / "server_logs"


def now_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _api_model_label(variant) -> str:
    if getattr(variant, "api_model", ""):
        return variant.api_model
    if getattr(variant, "backend", variant.fmt) == "openai-compatible":
        return variant.model_id
    if getattr(variant, "backend", variant.fmt) == "mlx":
        return variant.path
    return variant.key


def _api_key(variant, override: str | None) -> str | None:
    if override:
        return override
    env_name = getattr(variant, "api_key_env", "") or "OPENAI_API_KEY"
    return os.environ.get(env_name)


def _terminal_model_label(api_model_label: str, override: str | None) -> str:
    if override:
        return override
    litellm_prefixes = (
        "openai/",
        "azure/",
        "anthropic/",
        "gemini/",
        "vertex_ai/",
        "deepseek/",
        "xai/",
        "groq/",
        "together_ai/",
        "openrouter/",
    )
    if api_model_label.startswith(litellm_prefixes):
        return api_model_label
    return f"openai/{api_model_label}"


def _split_task_ids(values: tuple[str, ...]) -> list[str]:
    return [part for value in values for token in value.split(",") for part in token.split() if part]


def _effective_n_tasks(
    task_ids: list[str],
    n_tasks: int | None,
    all_tasks: bool,
) -> int | None:
    if all_tasks:
        return None
    if n_tasks is not None:
        return n_tasks
    return None if task_ids else 1


@click.command()
@click.option("--variant", required=True, help="Registry variant key to evaluate.")
@click.option("--dataset", default=DEFAULT_DATASET, show_default=True, help="Terminal-Bench dataset.")
@click.option(
    "--task-id",
    "task_ids",
    multiple=True,
    help="Terminal-Bench task id. Repeat or comma-separate values.",
)
@click.option(
    "--n-tasks",
    type=int,
    default=None,
    help="Number of tasks to sample. Defaults to 1 unless --task-id or --all-tasks is set.",
)
@click.option("--all-tasks", is_flag=True, help="Run every task in the selected dataset/filter.")
@click.option("--agent", default=DEFAULT_AGENT, show_default=True, help="Terminal-Bench agent.")
@click.option(
    "--terminal-bench-bin",
    default=DEFAULT_BIN,
    show_default=True,
    help="Terminal-Bench command. Quoted commands like 'uvx --from terminal-bench tb' work.",
)
@click.option(
    "--model-label",
    default=None,
    help="LiteLLM model label for Terminal-Bench. Defaults to openai/<registry model label>.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=EVAL_RESULTS_DIR,
    show_default=True,
    help="Root eval_scores directory for imported llm-bench results.",
)
@click.option("--port", type=int, default=9090, show_default=True)
@click.option("--context-size", type=int, default=16384, show_default=True)
@click.option("--docker-host", default=None, help="Optional DOCKER_HOST override.")
@click.option("--api-key", default=None, help="Optional API key override.")
@click.option("--run-id", default=None, help="Optional raw Terminal-Bench run id.")
@click.option("--agent-timeout-sec", type=int, default=600, show_default=True)
@click.option("--test-timeout-sec", type=int, default=300, show_default=True)
@click.option("--cleanup/--no-cleanup", default=True, show_default=True)
def main(
    variant: str,
    dataset: str,
    task_ids: tuple[str, ...],
    n_tasks: int | None,
    all_tasks: bool,
    agent: str,
    terminal_bench_bin: str,
    model_label: str | None,
    output_dir: Path,
    port: int,
    context_size: int,
    docker_host: str | None,
    api_key: str | None,
    run_id: str | None,
    agent_timeout_sec: int,
    test_timeout_sec: int,
    cleanup: bool,
) -> None:
    try:
        v = get_registry().variant(variant)
    except KeyError as exc:
        raise click.BadParameter(str(exc), param_hint="--variant") from exc
    if is_speed_only_variant(v):
        raise click.ClickException(f"{variant} is speed-only and cannot run evals.")

    safe_ts = now_safe()
    task_id_list = _split_task_ids(task_ids)
    run_dir = output_dir / f"{safe_ts}_{variant}_full" / TASK_NAME
    server_log = SERVER_LOG_DIR / f"{safe_ts}_{variant}_terminal_bench.log"
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    api_model_label = _api_model_label(v)
    tb_model_label = _terminal_model_label(api_model_label, model_label)
    click.echo(f"Terminal-Bench model: {tb_model_label}")
    click.echo(f"Dataset: {dataset}")
    if task_id_list:
        click.echo(f"Task ids: {', '.join(task_id_list)}")
    else:
        click.echo(f"Task count: {'all' if all_tasks else _effective_n_tasks([], n_tasks, False)}")

    t0 = time.perf_counter()
    try:
        with ModelServer(
            fmt=v.fmt,
            backend=getattr(v, "backend", v.fmt),
            artifact_type=getattr(v, "artifact_type", ""),
            model_path=v.resolved_path,
            port=port,
            context_size=context_size,
            log_file=server_log,
            runtime_root=getattr(v, "runtime_root", ""),
        ) as base_url:
            click.echo(f"Server: {base_url} (booted in {time.perf_counter() - t0:.1f}s)")
            result = run_terminal_bench(
                base_url=base_url,
                model_label=tb_model_label,
                output_dir=run_dir,
                dataset=dataset,
                agent=agent,
                task_ids=task_id_list,
                n_tasks=_effective_n_tasks(task_id_list, n_tasks, all_tasks),
                terminal_bench_bin=terminal_bench_bin,
                docker_host=docker_host,
                api_key=_api_key(v, api_key),
                run_id=run_id or f"{safe_ts}_{variant}",
                global_agent_timeout_sec=agent_timeout_sec,
                global_test_timeout_sec=test_timeout_sec,
                cleanup=cleanup,
            )
    except Exception as exc:
        click.echo(f"Terminal-Bench aborted: {exc}", err=True)
        sys.exit(1)

    if "error" in result:
        click.echo(f"Terminal-Bench failed: {result['error']}", err=True)
        if result.get("log"):
            click.echo(f"Log: {result['log']}", err=True)
        sys.exit(1)

    metrics = result["results"][TASK_NAME]
    click.echo(f"Imported llm-bench result: {result['results_file']}")
    click.echo(f"Raw Terminal-Bench run: {result['raw_run_dir']}")
    click.echo(
        "Metrics: "
        f"resolved={metrics['resolved_rate,none']:.4f}, "
        f"tasks={metrics['n_tasks,none']}, "
        f"avg_wall_s={metrics['avg_wall_s,none']:.1f}"
    )


if __name__ == "__main__":
    main()
