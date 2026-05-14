"""Pre-flight sanity check for the eval matrix.

Runs every suite task at limit=2 against one variant and flags configurations
that would silently produce all-zero scores. Catches:
  - extract filters that don't match the model's output format
  - generation truncation (max_gen_toks too low for the task)
  - chat template not being applied (tasks all return 0)
  - stop-token mismatches between server and harness

Exit code 0 = all tasks produced plausible numbers, 1 = at least one task
returned all zeros across every metric (likely a config bug, not a model
weakness — even random guessing on MMLU/HellaSwag scores >0).

Run before kicking off the multi-hour overnight matrix:

    uv run python scripts/preflight.py --variant 26B-MoE-mlx-4bit
    uv run python scripts/preflight.py  # picks the smallest local variant

Designed to finish in <10 min on the smallest variant.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import click

from llm_bench.evals import ModelServer, full_suite
from llm_bench.evals.lmeval import run_lmeval
from llm_bench.evals.suites import capabilities_for_backend, is_chat_task, supports_capabilities
from llm_bench.registry import Variant, get_registry

ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_DIR = ROOT / "results" / "preflight"

# Tasks where 0.0 is genuinely possible at limit=2 even with a good model
# (small sample, hard task) — don't alarm on those.
ZERO_OK_TASKS: set[str] = {
    "longbench",   # 21 subtasks, limit=2 only hits one or two
    "toxigen",     # binary detection, 2/2 correct or 0/2 — coin flip variance
}

PER_TASK_TIMEOUT_S = 10 * 60  # 10min ceiling per smoke task


def _smallest_local(registry) -> Variant | None:
    """Pick the smallest variant that's actually downloaded — fastest preflight."""
    candidates = [v for v in registry.variants if v.exists_locally()]
    if not candidates:
        return None
    # Heuristic: lower tier (4bit < 8bit) and smaller model = smaller bytes.
    return sorted(candidates, key=lambda v: (v.tier or "", v.model_id))[0]


def _all_zero(results: dict, task: str) -> bool:
    """True if every numeric metric in the task's results is exactly 0.0."""
    payload = results.get("results", {})
    if not isinstance(payload, dict):
        return False
    block = payload.get(task) or next(iter(payload.values()), {})
    if not isinstance(block, dict):
        return False
    nums = [v for k, v in block.items()
            if isinstance(v, (int, float)) and "stderr" not in k]
    return bool(nums) and all(v == 0.0 for v in nums)


@click.command()
@click.option("--variant", "variant_key", default=None,
              help="Registry key. Default: smallest locally-present variant.")
@click.option("--port", type=int, default=9091, show_default=True,
              help="Use a different port than the matrix run to avoid contention.")
@click.option("--limit", type=int, default=2, show_default=True)
def main(variant_key: str | None, port: int, limit: int) -> None:
    registry = get_registry()
    v = registry.variant(variant_key) if variant_key else _smallest_local(registry)
    if v is None:
        click.echo("→ no local variants. Run sync_models.py first.", err=True)
        sys.exit(2)
    if not v.exists_locally():
        click.echo(f"→ {v.key} not local. Run sync_models.py --variant {v.key}.", err=True)
        sys.exit(2)

    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = PREFLIGHT_DIR / f"{v.key}.server.log"
    out_root = PREFLIGHT_DIR / v.key

    caps = getattr(v, "capabilities", capabilities_for_backend(getattr(v, "backend", v.fmt)))
    tasks = [(d, t) for d, t in full_suite() if supports_capabilities(t, caps)]
    click.echo(f"→ preflight {v.key} ({v.fmt}, {v.quant}) "
               f"on {len(tasks)} tasks @ limit={limit}\n")

    suspects: list[str] = []
    failures: list[str] = []
    t_start = time.perf_counter()
    with ModelServer(
        fmt=v.fmt,
        backend=getattr(v, "backend", v.fmt),
        artifact_type=getattr(v, "artifact_type", ""),
        model_path=v.resolved_path,
        port=port,
        log_file=log_file,
        runtime_root=getattr(v, "runtime_root", ""),
    ) as base_url:
        api_label = getattr(v, "api_model_label", v.path if v.fmt == "mlx" else v.key)
        api_key = _api_key(v)
        for dim, task in tasks:
            click.echo(f"  [{dim}] {task:30s} ", nl=False)
            t0 = time.perf_counter()
            res = run_lmeval(
                task=task, base_url=base_url, model_label=api_label,
                output_dir=out_root / task, limit=limit,
                use_chat=is_chat_task(task), timeout_s=PER_TASK_TIMEOUT_S,
                api_key=api_key,
            )
            dt = time.perf_counter() - t0
            if "error" in res:
                click.echo(f"FAIL ({dt:.0f}s): {res['error']}")
                failures.append(task)
                continue
            if _all_zero(res, task) and task not in ZERO_OK_TASKS:
                click.echo(f"SUSPECT ({dt:.0f}s): all metrics 0.0")
                suspects.append(task)
            else:
                click.echo(f"OK ({dt:.0f}s)")

    click.echo(f"\n→ preflight done in {time.perf_counter() - t_start:.0f}s")
    if failures:
        click.echo(f"  ✗ {len(failures)} hard failures: {failures}", err=True)
    if suspects:
        click.echo(f"  ⚠ {len(suspects)} all-zero suspects: {suspects}", err=True)
        click.echo("    These tasks need filter/gen_kwargs review before the matrix run.", err=True)

    if failures or suspects:
        sys.exit(1)
    click.echo("  ✓ all tasks produced plausible numbers")


def _api_key(variant: Variant) -> str | None:
    env_name = getattr(variant, "api_key_env", "") or "OPENAI_API_KEY"
    return os.environ.get(env_name)


if __name__ == "__main__":
    main()
