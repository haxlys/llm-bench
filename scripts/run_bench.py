"""Run the speed/memory benchmark matrix for one or more variants.

Variants are read from models/registry.yaml — adding a new model is a YAML
edit, no code change. Idempotent: --skip-existing skips (variant, scenario,
bench_version) combos that already have N runs.

Examples:
    # Two variants, full matrix, skip already-measured combos
    uv run python scripts/run_bench.py \
        --variant 26B-MoE-mlx-4bit --variant 26B-MoE-gguf-q4 --skip-existing

    # All variants in registry that are missing data
    uv run python scripts/run_bench.py --all-pending

    # Single smoke scenario
    uv run python scripts/run_bench.py --variant 26B-MoE-mlx-8bit --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench import BENCH_VERSION  # noqa: E402
from llm_bench.aggregate import write_summary  # noqa: E402
from llm_bench.manifest import speed_is_measured, speed_manifest  # noqa: E402
from llm_bench.registry import Variant, get_registry  # noqa: E402
from llm_bench.runners import GGUFRunner, MLXRunner  # noqa: E402
from llm_bench.runners.base import write_raw  # noqa: E402
from llm_bench.scenarios import default_scenarios, smoke_scenarios  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
SUMMARY_CSV = ROOT / "results" / "summary.csv"


def _build_runner(variant: Variant):
    if variant.fmt == "mlx":
        return MLXRunner(model_id=variant.model_id, model_path=variant.resolved_path,
                         quant=variant.quant, variant_key=variant.key)
    return GGUFRunner(model_id=variant.model_id, model_path=variant.resolved_path,
                      quant=variant.quant, variant_key=variant.key)


def _resolve_targets(variant_keys: tuple, all_pending: bool, scenarios) -> list[Variant]:
    registry = get_registry()
    if variant_keys:
        return [registry.variant(k) for k in variant_keys]
    if all_pending:
        # Variants that are present locally + missing some scenarios
        manifest = speed_manifest(RAW_DIR)
        targets: list[Variant] = []
        for v in registry.variants:
            if not v.exists_locally():
                continue
            for sc in scenarios:
                if not speed_is_measured(manifest, v.key, sc.name, n_required=3):
                    targets.append(v)
                    break
        return targets
    return []


@click.command()
@click.option("--variant", multiple=True, help="Registry key (repeatable)")
@click.option("--all-pending", is_flag=True,
              help="Run every variant in registry that has missing scenarios")
@click.option("--runs", default=3, show_default=True)
@click.option("--warmup/--no-warmup", default=True)
@click.option("--smoke/--full", default=False, help="Smoke = single scenario")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip combos already measured at this bench_version")
def main(variant: tuple, all_pending: bool, runs: int, warmup: bool,
         smoke: bool, skip_existing: bool):
    scenarios = smoke_scenarios() if smoke else default_scenarios()
    targets = _resolve_targets(variant, all_pending, scenarios)
    if not targets:
        click.echo("→ no targets. Specify --variant or --all-pending.")
        sys.exit(2)

    manifest = speed_manifest(RAW_DIR) if skip_existing else {}
    click.echo(f"→ {len(targets)} variant(s), {len(scenarios)} scenarios, "
               f"bench_version={BENCH_VERSION}")
    for v in targets:
        click.echo(f"  · {v.key} ({v.fmt}, {v.quant})")

    for v in targets:
        runner = _build_runner(v)
        click.echo(f"\n=== {v.key} ===")
        for sc in scenarios:
            if skip_existing and speed_is_measured(manifest, v.key, sc.name, n_required=runs):
                click.echo(f"  [skip] {sc.name} (already has {runs} runs at v{BENCH_VERSION})")
                continue
            if warmup:
                click.echo(f"  [warmup] {sc.name}")
                try:
                    runner.run(sc, run_idx=0)
                except Exception as e:
                    click.echo(f"    ! warmup failed: {e}", err=True); continue
            for i in range(1, runs + 1):
                click.echo(f"  [run {i}] {sc.name} ", nl=False)
                try:
                    res = runner.run(sc, run_idx=i)
                    write_raw(res, RAW_DIR)
                    click.echo(f"pp={res.pp_tps:.1f} tg={res.tg_tps:.1f} "
                               f"mem={res.peak_mem_gb:.1f}GB")
                except Exception as e:
                    click.echo(f"FAILED: {e}", err=True)

    out = write_summary(RAW_DIR, SUMMARY_CSV)
    click.echo(f"\n→ summary written to {out}")


if __name__ == "__main__":
    main()
