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
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from llm_bench.evals import ModelServer, full_suite, smoke_suite
from llm_bench.evals.bfcl import run_bfcl
from llm_bench.evals.lmeval import run_lmeval
from llm_bench.evals.suites import is_chat_task, supports_fmt
from llm_bench.manifest import eval_is_measured, eval_manifest
from llm_bench.registry import Variant, get_registry

ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = ROOT / "results" / "eval_scores"
SERVER_LOG_DIR = ROOT / "results" / "server_logs"


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


@click.command()
@click.option("--variant", multiple=True, help="Registry key (repeatable)")
@click.option("--all-variants", is_flag=True, help="Every locally-present variant")
@click.option("--suite", type=click.Choice(["smoke", "full"]), default="smoke", show_default=True)
@click.option("--limit", type=int, default=None,
              help="Override per-task sample limit. Smoke default = 3.")
@click.option("--port", type=int, default=9090, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("--include-bfcl", is_flag=True)
@click.option("--bfcl-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
def main(variant: tuple, all_variants: bool, suite: str, limit: int | None,
         port: int, skip_existing: bool, include_bfcl: bool, bfcl_dir: Path | None):
    targets = _resolve_targets(variant, all_variants)
    if not targets:
        click.echo("→ no targets. Specify --variant or --all-variants.")
        sys.exit(2)

    tasks = smoke_suite() if suite == "smoke" else full_suite()
    effective_limit = limit if limit is not None else (3 if suite == "smoke" else None)

    measured = eval_manifest(EVAL_RESULTS_DIR).measured if skip_existing else set()

    click.echo(f"→ {len(targets)} variants × {len(tasks)} tasks "
               f"(suite={suite}, limit={effective_limit})")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    grand_summary: list[dict] = []
    for v in targets:
        run_id = f"{now_safe()}_{v.key}_{suite}"
        out_dir = EVAL_RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        server_log = SERVER_LOG_DIR / f"{run_id}.log"

        click.echo(f"\n=== {v.key} ({v.fmt}, {v.quant}) ===")

        # Pre-filter tasks for this variant: skip unsupported, skip already-measured
        runnable: list[tuple[str, str]] = []
        for dim, task in tasks:
            if not supports_fmt(task, v.fmt):
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

        if not runnable and not include_bfcl:
            click.echo(f"  → all tasks already measured / unsupported, skipping server boot")
            continue

        t0 = time.perf_counter()
        try:
            with ModelServer(fmt=v.fmt, model_path=v.resolved_path, port=port,
                             log_file=server_log) as base_url:
                click.echo(f"  server: {base_url} (booted in {time.perf_counter()-t0:.1f}s)")
                api_model_label = v.path if v.fmt == "mlx" else v.key
                for dim, task in runnable:
                    click.echo(f"  [{dim}] {task} ", nl=False)
                    use_chat = is_chat_task(task)
                    ts0 = time.perf_counter()
                    res = run_lmeval(
                        task=task, base_url=base_url, model_label=api_model_label,
                        output_dir=out_dir / task, limit=effective_limit,
                        use_chat=use_chat,
                    )
                    dt = time.perf_counter() - ts0
                    if "error" in res:
                        click.echo(f"FAIL ({dt:.0f}s): {res['error']}")
                    else:
                        click.echo(f"OK ({dt:.0f}s)")
                    grand_summary.append({
                        "variant": v.key, "model_id": v.model_id, "fmt": v.fmt,
                        "quant": v.quant, "dim": dim, "task": task,
                        "wall_s": round(dt, 1), **res,
                    })
                if include_bfcl:
                    click.echo(f"  [tool] bfcl ", nl=False)
                    res = run_bfcl(base_url, v.key, out_dir / "bfcl",
                                   bfcl_dir, effective_limit)
                    click.echo(res.get("status", res.get("error", "OK")))
                    grand_summary.append({"variant": v.key, "dim": "tool",
                                          "task": "bfcl", **res})
        except Exception as e:
            click.echo(f"  ! variant {v.key} aborted: {e}", err=True)
            grand_summary.append({"variant": v.key, "fatal": str(e)})

    summary_path = EVAL_RESULTS_DIR / f"summary_{now_safe()}_{suite}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(grand_summary, indent=2, ensure_ascii=False))
    click.echo(f"\n→ summary: {summary_path}")


if __name__ == "__main__":
    main()
