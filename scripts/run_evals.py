"""Run multi-dimensional evals for one or more variants.

Variants are resolved from models/registry.yaml. Boots a per-variant
OpenAI-compatible server, runs the requested suite via lm-eval-harness,
tears down. Idempotent: --skip-existing skips (variant, task) pairs that
already have a non-empty results JSON.

Examples:
    # Smoke: single variant, limit=2
    uv run python scripts/run_evals.py --variant 26B-MoE-mlx-8bit \
        --suite smoke --limit 2

    # Full overnight: all variants in registry, skip already-measured
    uv run python scripts/run_evals.py --all-variants --suite full
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from llm_bench.evals import ModelServer, full_suite, smoke_suite
from llm_bench.evals.bfcl import run_bfcl
from llm_bench.evals.bigcodebench_runner import bigcodebench_available
from llm_bench.evals.bigcodebench_runner import run_bigcodebench_hard
from llm_bench.evals.ifeval_resilient_runner import run_leaderboard_ifeval_resilient
from llm_bench.evals.evalplus_runner import run_evalplus
from llm_bench.evals.kmmlu_pro_runner import kmmlu_pro_available
from llm_bench.evals.kmmlu_pro_runner import run_kmmlu_pro
from llm_bench.evals.livecodebench_runner import DEFAULT_RELEASE as LCB_RELEASE
from llm_bench.evals.livecodebench_runner import livecodebench_available
from llm_bench.evals.livecodebench_runner import run_livecodebench
from llm_bench.evals.livebench_runner import DEFAULT_RELEASE as LIVEBENCH_RELEASE
from llm_bench.evals.livebench_runner import livebench_available
from llm_bench.evals.livebench_runner import run_livebench
from llm_bench.evals.lmeval import run_lmeval
from llm_bench.evals.memory_stability import run_memory_stability
from llm_bench.evals.sourceqa import DEFAULT_TASKS_PATH as SOURCEQA_DEFAULT_TASKS
from llm_bench.evals.sourceqa import run_sourceqa
from llm_bench.evals.suites import (
    capabilities_for_backend,
    external_suite,
    external_supports_capabilities,
    is_chat_task,
    long_suite,
    supports_capabilities,
    task_lane,
)
from llm_bench.evals.terminal_bench_runner import DEFAULT_AGENT as TERMINAL_BENCH_AGENT
from llm_bench.evals.terminal_bench_runner import DEFAULT_BIN as TERMINAL_BENCH_BIN
from llm_bench.evals.terminal_bench_runner import DEFAULT_DATASET as TERMINAL_BENCH_DATASET
from llm_bench.evals.terminal_bench_runner import run_terminal_bench
from llm_bench.evals.terminal_bench_runner import terminal_bench_available
from llm_bench.evals.trace import append_trace
from llm_bench.manifest import eval_is_measured, eval_manifest
from llm_bench.registry import Variant, get_registry, is_speed_only_variant

ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = ROOT / "results" / "eval_scores"
SERVER_LOG_DIR = ROOT / "results" / "server_logs"
TRACE_DIR = ROOT / "results" / "eval_traces"


def now_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_targets(variant_keys: tuple, all_variants: bool) -> list[Variant]:
    import click
    registry = get_registry()
    if variant_keys:
        return [registry.variant(k) for k in variant_keys]
    if all_variants:
        targets = []
        for v in registry.variants:
            if v.exists_locally():
                targets.append(v)
            else:
                click.echo(f"  · {v.key}: not present locally, skipping. "
                           f"Run sync_models.py --variant {v.key} first.", err=True)
        return targets
    return []


def _filter_eval_targets(targets: list[Variant]) -> list[Variant]:
    eval_targets = []
    for variant in targets:
        if is_speed_only_variant(variant):
            click.echo(
                f"  · {variant.key}: speed-only MTPLX variant, "
                "skipping eval server path.",
                err=True,
            )
            continue
        eval_targets.append(variant)
    return eval_targets


def _trace_task_result(
    trace_path: Path,
    variant_key: str,
    task: str,
    runner: str,
    dim: str,
    wall_s: float,
    result: dict,
) -> Path:
    """Append one task execution outcome to the per-run JSONL trace."""
    status = "error" if result.get("error") else "ok"
    return append_trace(trace_path, {
        "variant": variant_key,
        "task": task,
        "runner": runner,
        "dim": dim,
        "wall_s": round(wall_s, 1),
        "status": status,
        "results_file": result.get("results_file"),
        "samples_file": result.get("samples_file"),
        "log": result.get("log"),
        "error": result.get("error"),
    })


def _variant_capabilities(variant: Variant) -> frozenset[str]:
    return getattr(
        variant,
        "capabilities",
        capabilities_for_backend(getattr(variant, "backend", variant.fmt)),
    )


def _api_model_label(variant: Variant) -> str:
    if getattr(variant, "api_model", ""):
        return variant.api_model
    if getattr(variant, "backend", variant.fmt) == "openai-compatible":
        return variant.model_id
    if getattr(variant, "backend", variant.fmt) == "mlx":
        return variant.path
    return variant.key


def _tokenizer_label(variant: Variant) -> str:
    if getattr(variant, "tokenizer", ""):
        return variant.tokenizer
    if getattr(variant, "api_model", ""):
        return variant.api_model
    if getattr(variant, "backend", variant.fmt) == "mlx":
        return variant.path
    return variant.model_id


def _api_key(variant: Variant) -> str | None:
    env_name = getattr(variant, "api_key_env", "") or "OPENAI_API_KEY"
    return os.environ.get(env_name)


def _server_context_size(tasks: list[tuple[str, str]]) -> int:
    if any(task == "longbench" for _, task in tasks):
        return 65536
    return 16384


def _selected_tasks(task_filters: tuple[str, ...]) -> set[str]:
    return {
        task.strip()
        for raw in task_filters
        for task in raw.split(",")
        if task.strip()
    }


def _filter_tasks(
    tasks: list[tuple[str, str]],
    selected_tasks: set[str],
) -> list[tuple[str, str]]:
    if not selected_tasks:
        return tasks
    return [(dim, task) for dim, task in tasks if task in selected_tasks]


def _available_tasks(
    tasks: list[tuple[str, str]],
    *,
    include_external: bool,
) -> set[str]:
    available = {task for _, task in tasks}
    if include_external:
        available.update(task for _, task, _ in external_suite())
    return available


def _external_skip_reason(runner: str, effective_limit: int | None) -> str | None:
    if runner == "evalplus" and effective_limit is not None:
        return "skipped_limit_incompatible"
    if runner == "livecodebench" and not livecodebench_available():
        return "skipped_unavailable_external"
    if runner == "bigcodebench" and not bigcodebench_available():
        return "skipped_unavailable_external"
    if runner == "livebench" and not livebench_available():
        return "skipped_unavailable_external"
    if runner == "kmmlu_pro" and not kmmlu_pro_available():
        return "skipped_unavailable_external"
    if runner == "terminal_bench" and not terminal_bench_available(_terminal_bench_bin()):
        return "skipped_unavailable_external"
    return None


def _coverage_done_statuses() -> set[str]:
    # "unsupported by this backend" tasks are intentional skips, not hard failures.
    return {
        "completed",
        "skipped_already_measured",
        "skipped_unsupported_external",
        "skipped_optional_disabled",
    }


def _build_coverage_row(dim: str, task: str, runner: str, status: str) -> dict:
    lane = task_lane(task)
    return {
        "dim": dim,
        "task": task,
        "runner": runner,
        "lane": lane,
        "required": lane == "primary",
        "status": status,
    }


def _set_coverage_status(
    rows: list[dict],
    dim: str,
    task: str,
    runner: str,
    status: str,
) -> None:
    """Mutate exactly one coverage row for the task/run tuple."""
    for row in rows:
        if row["dim"] == dim and row["task"] == task and row["runner"] == runner:
            row["status"] = status
            return
    rows.append(_build_coverage_row(dim, task, runner, status))


def _coverage_summary(rows: list[dict]) -> dict:
    done = _coverage_done_statuses()
    required_rows = [row for row in rows if row.get("required", True)]
    missing_rows = [row for row in required_rows if row["status"] not in done]
    optional_rows = [row for row in rows if not row.get("required", True)]
    return {
        "required": len(required_rows),
        "completed": len(required_rows) - len(missing_rows),
        "missing_count": len(missing_rows),
        "missing": missing_rows,
        "optional": len(optional_rows),
    }


def _livecodebench_release() -> str:
    return os.environ.get("LIVE_CODE_BENCH_RELEASE", LCB_RELEASE)


def _terminal_bench_dataset() -> str:
    return os.environ.get("TERMINAL_BENCH_DATASET", TERMINAL_BENCH_DATASET)


def _terminal_bench_agent() -> str:
    return os.environ.get("TERMINAL_BENCH_AGENT", TERMINAL_BENCH_AGENT)


def _terminal_bench_bin() -> str:
    return os.environ.get("TERMINAL_BENCH_BIN", TERMINAL_BENCH_BIN)


def _terminal_bench_model_label(api_model_label: str) -> str:
    override = os.environ.get("TERMINAL_BENCH_MODEL", "").strip()
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


def _terminal_bench_task_ids() -> list[str]:
    raw = os.environ.get("TERMINAL_BENCH_TASK_IDS", "")
    return [part for token in raw.split(",") for part in token.split() if part]


def _terminal_bench_n_tasks(task_ids: list[str], effective_limit: int | None) -> int | None:
    raw = os.environ.get("TERMINAL_BENCH_N_TASKS", "").strip()
    if raw:
        return int(raw)
    if task_ids:
        return None
    if os.environ.get("TERMINAL_BENCH_FULL", "").lower() in {"1", "true", "yes"}:
        return None
    return effective_limit if effective_limit is not None else 1


def _terminal_bench_timeout(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _lmeval_runner_for_task(task: str, resilient_ifeval: bool) -> str:
    if task == "leaderboard_ifeval" and resilient_ifeval:
        return "ifeval_resilient"
    return "lm-eval"


@click.command()
@click.option("--variant", multiple=True, help="Registry key (repeatable)")
@click.option("--all-variants", is_flag=True, help="Every locally-present variant")
@click.option("--suite", type=click.Choice(["smoke", "full", "long"]), default="smoke", show_default=True)
@click.option("--task", "task_filter", multiple=True,
              help="Run only these task ids. Repeat or comma-separate values.")
@click.option("--limit", type=int, default=None,
              help="Override per-task sample limit. Smoke default = 3.")
@click.option("--port", type=int, default=9090, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("--include-bfcl", is_flag=True,
              help="Run BFCL v4 tool-use eval (requires `uv pip install bfcl-eval`, ~30 min/variant).")
@click.option("--include-terminal-bench", is_flag=True,
              help="Deprecated no-op: Terminal-Bench is a primary full-suite task.")
@click.option("--resilient-ifeval/--strict-ifeval", default=False,
              help="Use resilient leaderboard_ifeval runner for per-sample API errors.")
@click.option("--strict-coverage", is_flag=True, default=False,
              help="Fail if any required task in this run has no successful result.")
@click.option("--sourceqa-tasks", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help=f"Source-grounding task YAML/JSON (default: {SOURCEQA_DEFAULT_TASKS}).")
@click.option("--sourceqa-judge-model", default=None,
              help="Optional judge model label for sourceqa metadata; deterministic score remains primary.")
def main(variant: tuple, all_variants: bool, suite: str, task_filter: tuple[str, ...],
         limit: int | None,
         port: int, skip_existing: bool, include_bfcl: bool,
         include_terminal_bench: bool, resilient_ifeval: bool,
         strict_coverage: bool, sourceqa_tasks: str | None, sourceqa_judge_model: str | None):
    targets = _filter_eval_targets(_resolve_targets(variant, all_variants))
    if not targets:
        click.echo("→ no targets. Specify --variant or --all-variants.")
        sys.exit(2)

    if suite == "smoke":
        tasks = smoke_suite()
    elif suite == "long":
        tasks = long_suite()
    else:
        tasks = full_suite()
    selected_tasks = _selected_tasks(task_filter)
    unknown_tasks = selected_tasks - _available_tasks(tasks, include_external=suite == "full")
    if unknown_tasks:
        raise click.BadParameter(
            "unknown task id(s): "
            + ", ".join(sorted(unknown_tasks))
            + ". External tasks require --suite full; ProgramBench is imported "
            "with scripts/import_programbench.py; Terminal-Bench can also run "
            "with scripts/run_terminal_bench.py."
        )
    tasks = _filter_tasks(tasks, selected_tasks)
    effective_limit = limit if limit is not None else (3 if suite == "smoke" else 1 if suite == "long" else None)
    server_context_size = _server_context_size(tasks)

    measured = eval_manifest(EVAL_RESULTS_DIR).measured if skip_existing else set()
    external_task_count = (
        sum(1 for _, task, _ in external_suite() if not selected_tasks or task in selected_tasks)
        if suite == "full"
        else 0
    )

    click.echo(f"→ {len(targets)} variants × {len(tasks) + external_task_count} tasks "
               f"(suite={suite}, limit={effective_limit})")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    grand_summary: list[dict] = []
    has_coverage_failure = False
    for v in targets:
        run_id = f"{now_safe()}_{v.key}_{suite}"
        out_dir = EVAL_RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        server_log = SERVER_LOG_DIR / f"{run_id}.log"
        trace_path = TRACE_DIR / f"{run_id}.jsonl"

        click.echo(f"\n=== {v.key} ({v.fmt}, {v.quant}) ===")

        # Pre-filter tasks for this variant: skip unsupported, skip already-measured
        variant_coverage: list[dict] = []
        runnable: list[tuple[str, str]] = []
        capabilities = _variant_capabilities(v)
        for dim, task in tasks:
            if not supports_capabilities(task, capabilities):
                grand_summary.append({
                    "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                    "quant": v.quant, "dim": dim, "task": task,
                    "status": "skipped_logprob",
                })
                continue
            if skip_existing and eval_is_measured(measured, v.key, task):
                grand_summary.append({
                    "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                    "quant": v.quant, "dim": dim, "task": task,
                    "status": "skipped_already_measured",
                })
                runner = _lmeval_runner_for_task(task, resilient_ifeval)
                variant_coverage.append(_build_coverage_row(dim, task, runner, "skipped_already_measured"))
                continue
            runnable.append((dim, task))
            runner = _lmeval_runner_for_task(task, resilient_ifeval)
            variant_coverage.append(_build_coverage_row(dim, task, runner, "pending"))

        external_runnable: list[tuple[str, str, str]] = []
        if suite == "full":
            for dim, task, runner in external_suite():
                if selected_tasks and task not in selected_tasks:
                    continue
                if runner == "bfcl" and not include_bfcl:
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": "skipped_bfcl_disabled",
                    })
                    variant_coverage.append(_build_coverage_row(dim, task, runner, "skipped_optional_disabled"))
                    continue
                if reason := _external_skip_reason(runner, effective_limit):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": reason,
                    })
                    variant_coverage.append(_build_coverage_row(dim, task, runner, reason))
                    continue
                if not external_supports_capabilities(task, runner, capabilities):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": "skipped_unsupported_external",
                    })
                    variant_coverage.append(_build_coverage_row(dim, task, runner, "skipped_unsupported_external"))
                    continue
                if skip_existing and eval_is_measured(measured, v.key, task):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": "skipped_already_measured",
                    })
                    variant_coverage.append(_build_coverage_row(dim, task, runner, "skipped_already_measured"))
                    continue
                external_runnable.append((dim, task, runner))
                variant_coverage.append(_build_coverage_row(dim, task, runner, "pending"))
        if not runnable and not external_runnable:
            click.echo("  → all tasks already measured / unsupported, skipping server boot")
            if variant_coverage:
                coverage = _coverage_summary(variant_coverage)
                click.echo(
                    f"  coverage: {coverage['completed']}/{coverage['required']} required tasks "
                    f"completed ({coverage['missing_count']} missing)"
                )
                if coverage["missing"]:
                    for row in coverage["missing"]:
                        click.echo(
                            f"    - {row['dim']}/{row['runner']}/{row['task']}: "
                            f"{row['status']}"
                        )
                has_coverage_failure = has_coverage_failure or (
                    strict_coverage and coverage["missing_count"] > 0
                )
                grand_summary.append({
                    "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                    "quant": v.quant, "status": "coverage_report",
                    "suite": suite, **coverage,
                })
            continue

        t0 = time.perf_counter()
        try:
            with ModelServer(
                fmt=v.fmt,
                backend=getattr(v, "backend", v.fmt),
                artifact_type=getattr(v, "artifact_type", ""),
                model_path=v.resolved_path,
                port=port,
                context_size=server_context_size,
                log_file=server_log,
                runtime_root=getattr(v, "runtime_root", ""),
            ) as base_url:
                click.echo(f"  server: {base_url} (booted in {time.perf_counter()-t0:.1f}s)")
                api_model_label = _api_model_label(v)
                tokenizer_label = _tokenizer_label(v)
                api_key = _api_key(v)
                for dim, task in runnable:
                    click.echo(f"  [{dim}] {task} ", nl=False)
                    use_chat = is_chat_task(task)
                    ts0 = time.perf_counter()
                    runner = _lmeval_runner_for_task(task, resilient_ifeval)
                    if task == "leaderboard_ifeval" and resilient_ifeval:
                        res = run_leaderboard_ifeval_resilient(
                            base_url=base_url,
                            model_label=api_model_label,
                            output_dir=out_dir / task,
                            limit=effective_limit,
                            api_key=api_key,
                        )
                    else:
                        res = run_lmeval(
                            task=task, base_url=base_url, model_label=api_model_label,
                            output_dir=out_dir / task, limit=effective_limit,
                            use_chat=use_chat,
                            tokenizer_label=tokenizer_label,
                            api_key=api_key,
                        )
                    dt = time.perf_counter() - ts0
                    if "error" in res:
                        click.echo(f"FAIL ({dt:.0f}s): {res['error']}")
                    else:
                        click.echo(f"OK ({dt:.0f}s)")
                    _trace_task_result(
                        trace_path=trace_path,
                        variant_key=v.key,
                        task=task,
                        runner=runner,
                        dim=dim,
                        wall_s=dt,
                        result=res,
                    )
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task, "runner": runner,
                        "wall_s": round(dt, 1), **res,
                    })
                    _set_coverage_status(
                        variant_coverage,
                        dim,
                        task,
                        runner,
                        "completed" if "error" not in res else "failed",
                    )
                # External runners talk to the same server but don't go through
                # lm-eval-harness.
                # Skip them on smoke runs because each is a heavyweight tool.
                # BFCL is gated on --include-bfcl (manual bfcl-eval install).
                if suite == "full":
                    for dim, task, runner in external_runnable:
                        click.echo(f"  [{dim}/{runner}] {task} ", nl=False)
                        ts0 = time.perf_counter()
                        if runner == "evalplus":
                            res = run_evalplus(
                                dataset=task, base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "livecodebench":
                            res = run_livecodebench(
                                release=_livecodebench_release(), base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "bigcodebench":
                            res = run_bigcodebench_hard(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "bfcl":
                            res = run_bfcl(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "sourceqa":
                            res = run_sourceqa(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                tasks_path=Path(sourceqa_tasks) if sourceqa_tasks else SOURCEQA_DEFAULT_TASKS,
                                limit=effective_limit,
                                judge_model=sourceqa_judge_model,
                                api_key=api_key,
                            )
                        elif runner == "memory_stability":
                            res = run_memory_stability(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "livebench":
                            res = run_livebench(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                release=os.environ.get("LIVEBENCH_RELEASE", LIVEBENCH_RELEASE),
                                api_key=api_key,
                            )
                        elif runner == "kmmlu_pro":
                            res = run_kmmlu_pro(
                                base_url=base_url,
                                model_label=api_model_label,
                                output_dir=out_dir / task,
                                limit=effective_limit,
                                api_key=api_key,
                            )
                        elif runner == "terminal_bench":
                            terminal_task_ids = _terminal_bench_task_ids()
                            res = run_terminal_bench(
                                base_url=base_url,
                                model_label=_terminal_bench_model_label(api_model_label),
                                output_dir=out_dir / task,
                                dataset=_terminal_bench_dataset(),
                                agent=_terminal_bench_agent(),
                                task_ids=terminal_task_ids,
                                n_tasks=_terminal_bench_n_tasks(
                                    terminal_task_ids,
                                    effective_limit,
                                ),
                                terminal_bench_bin=_terminal_bench_bin(),
                                docker_host=(
                                    os.environ.get("TERMINAL_BENCH_DOCKER_HOST")
                                    or os.environ.get("DOCKER_HOST")
                                ),
                                api_key=api_key,
                                global_agent_timeout_sec=_terminal_bench_timeout(
                                    "TERMINAL_BENCH_AGENT_TIMEOUT_SEC",
                                    600,
                                ),
                                global_test_timeout_sec=_terminal_bench_timeout(
                                    "TERMINAL_BENCH_TEST_TIMEOUT_SEC",
                                    300,
                                ),
                            )
                        else:
                            # Surface newly declared-but-unimplemented runners in
                            # the summary instead of silently ignoring them.
                            res = {"task": task, "error": f"runner not implemented: {runner}"}
                        dt = time.perf_counter() - ts0
                        if "error" in res:
                            click.echo(f"FAIL ({dt:.0f}s): {res['error']}")
                        else:
                            click.echo(f"OK ({dt:.0f}s)")
                        _trace_task_result(
                            trace_path=trace_path,
                            variant_key=v.key,
                            task=task,
                            runner=runner,
                            dim=dim,
                            wall_s=dt,
                            result=res,
                        )
                        grand_summary.append({
                            "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                            "quant": v.quant, "dim": dim, "task": task,
                            "runner": runner, "wall_s": round(dt, 1), **res,
                        })
                        _set_coverage_status(
                            variant_coverage,
                            dim,
                            task,
                            runner,
                            "completed" if "error" not in res else "failed",
                        )
        except Exception as e:
            click.echo(f"  ! variant {v.key} aborted: {e}", err=True)
            grand_summary.append({"variant": v.key, "fatal": str(e)})
            for row in variant_coverage:
                if row["status"] == "pending":
                    row["status"] = "failed"

        if variant_coverage:
            coverage = _coverage_summary(variant_coverage)
            click.echo(
                f"  coverage: {coverage['completed']}/{coverage['required']} required tasks "
                f"completed ({coverage['missing_count']} missing)"
            )
            if coverage["missing"]:
                for row in coverage["missing"]:
                    click.echo(
                        f"    - {row['dim']}/{row['runner']}/{row['task']}: {row['status']}"
                    )
            has_coverage_failure = has_coverage_failure or (
                strict_coverage and coverage["missing_count"] > 0
            )
            grand_summary.append({
                "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                "quant": v.quant, "status": "coverage_report",
                "suite": suite, **coverage,
            })

    summary_path = EVAL_RESULTS_DIR / f"summary_{now_safe()}_{suite}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(grand_summary, indent=2, ensure_ascii=False))
    click.echo(f"\n→ summary: {summary_path}")
    if has_coverage_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
