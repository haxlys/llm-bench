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
from llm_bench.evals.evalplus_runner import run_evalplus
from llm_bench.evals.livecodebench_runner import DEFAULT_RELEASE as LCB_RELEASE
from llm_bench.evals.livecodebench_runner import livecodebench_available
from llm_bench.evals.livecodebench_runner import run_livecodebench
from llm_bench.evals.lmeval import run_lmeval
from llm_bench.evals.sourceqa import DEFAULT_TASKS_PATH as SOURCEQA_DEFAULT_TASKS
from llm_bench.evals.sourceqa import run_sourceqa
from llm_bench.evals.suites import (
    capabilities_for_backend,
    external_suite,
    external_supports_capabilities,
    is_chat_task,
    long_suite,
    supports_capabilities,
)
from llm_bench.evals.trace import append_trace
from llm_bench.manifest import eval_is_measured, eval_manifest
from llm_bench.registry import Variant, get_registry

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


def _external_skip_reason(runner: str, effective_limit: int | None) -> str | None:
    if runner == "evalplus" and effective_limit is not None:
        return "skipped_limit_incompatible"
    if runner == "livecodebench" and not livecodebench_available():
        return "skipped_unavailable_external"
    return None


@click.command()
@click.option("--variant", multiple=True, help="Registry key (repeatable)")
@click.option("--all-variants", is_flag=True, help="Every locally-present variant")
@click.option("--suite", type=click.Choice(["smoke", "full", "long"]), default="smoke", show_default=True)
@click.option("--limit", type=int, default=None,
              help="Override per-task sample limit. Smoke default = 3.")
@click.option("--port", type=int, default=9090, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("--include-bfcl", is_flag=True,
              help="Run BFCL v4 tool-use eval (requires `uv pip install bfcl-eval`, ~30 min/variant).")
@click.option("--sourceqa-tasks", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help=f"Source-grounding task YAML/JSON (default: {SOURCEQA_DEFAULT_TASKS}).")
@click.option("--sourceqa-judge-model", default=None,
              help="Optional judge model label for sourceqa metadata; deterministic score remains primary.")
def main(variant: tuple, all_variants: bool, suite: str, limit: int | None,
         port: int, skip_existing: bool, include_bfcl: bool,
         sourceqa_tasks: str | None, sourceqa_judge_model: str | None):
    targets = _resolve_targets(variant, all_variants)
    if not targets:
        click.echo("→ no targets. Specify --variant or --all-variants.")
        sys.exit(2)

    if suite == "smoke":
        tasks = smoke_suite()
    elif suite == "long":
        tasks = long_suite()
    else:
        tasks = full_suite()
    effective_limit = limit if limit is not None else (3 if suite == "smoke" else 1 if suite == "long" else None)
    server_context_size = _server_context_size(tasks)

    measured = eval_manifest(EVAL_RESULTS_DIR).measured if skip_existing else set()

    click.echo(f"→ {len(targets)} variants × {len(tasks)} tasks "
               f"(suite={suite}, limit={effective_limit})")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    grand_summary: list[dict] = []
    for v in targets:
        run_id = f"{now_safe()}_{v.key}_{suite}"
        out_dir = EVAL_RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        server_log = SERVER_LOG_DIR / f"{run_id}.log"
        trace_path = TRACE_DIR / f"{run_id}.jsonl"

        click.echo(f"\n=== {v.key} ({v.fmt}, {v.quant}) ===")

        # Pre-filter tasks for this variant: skip unsupported, skip already-measured
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
                continue
            runnable.append((dim, task))

        external_runnable: list[tuple[str, str, str]] = []
        if suite == "full":
            for dim, task, runner in external_suite():
                if runner == "bfcl" and not include_bfcl:
                    continue
                if reason := _external_skip_reason(runner, effective_limit):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": reason,
                    })
                    continue
                if not external_supports_capabilities(task, runner, capabilities):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": "skipped_unsupported_external",
                    })
                    continue
                if skip_existing and eval_is_measured(measured, v.key, task):
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "runner": runner, "status": "skipped_already_measured",
                    })
                    continue
                external_runnable.append((dim, task, runner))
        if not runnable and not external_runnable:
            click.echo("  → all tasks already measured / unsupported, skipping server boot")
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
            ) as base_url:
                click.echo(f"  server: {base_url} (booted in {time.perf_counter()-t0:.1f}s)")
                api_model_label = _api_model_label(v)
                tokenizer_label = _tokenizer_label(v)
                api_key = _api_key(v)
                for dim, task in runnable:
                    click.echo(f"  [{dim}] {task} ", nl=False)
                    use_chat = is_chat_task(task)
                    ts0 = time.perf_counter()
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
                        runner="lm-eval",
                        dim=dim,
                        wall_s=dt,
                        result=res,
                    )
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "wall_s": round(dt, 1), **res,
                    })
                # External runners (EvalPlus, LiveCodeBench, BFCL) — they
                # talk to the same server but don't go through lm-eval-harness.
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
                                release=LCB_RELEASE, base_url=base_url,
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
                        else:
                            # simple_evals and kmmlu_pro runners not yet wired.
                            # Surface the gap explicitly so it shows in summary.
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
        except Exception as e:
            click.echo(f"  ! variant {v.key} aborted: {e}", err=True)
            grand_summary.append({"variant": v.key, "fatal": str(e)})

    summary_path = EVAL_RESULTS_DIR / f"summary_{now_safe()}_{suite}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(grand_summary, indent=2, ensure_ascii=False))
    click.echo(f"\n→ summary: {summary_path}")


if __name__ == "__main__":
    main()
