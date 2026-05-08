"""Registry-driven model download.

Reads models/registry.yaml and ensures every variant's local artifact is
present. Idempotent — already-present models are skipped.

Examples:
    # Download every missing variant
    uv run python scripts/sync_models.py --all-missing

    # Specific variant
    uv run python scripts/sync_models.py --variant 26B-MoE-gguf-q8

    # All variants of a logical model
    uv run python scripts/sync_models.py --model gemma-4-31B-it

    # Just check status (no downloads)
    uv run python scripts/sync_models.py --check
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click

from llm_bench.registry import Variant, get_registry


def hf_download(variant: Variant) -> int:
    """Run `hf download` for the given variant. Returns subprocess rc."""
    if not shutil.which("hf"):
        click.echo("ERROR: hf CLI not found. Install: uv tool install huggingface_hub",
                   err=True)
        return 1
    if variant.fmt == "gguf":
        if variant.download is None:
            click.echo(f"  no download spec for gguf variant {variant.key}; skipping",
                       err=True)
            return 0
        local_dir = Path(variant.resolved_path).parent
        if variant.download.pattern and "/" in variant.download.pattern:
            first_segment = variant.download.pattern.split("/", 1)[0]
            if local_dir.name == first_segment:
                local_dir = local_dir.parent
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["hf", "download", variant.download.repo,
               "--local-dir", str(local_dir)]
        if variant.download.pattern:
            cmd.extend(["--include", variant.download.pattern])
        if variant.download.revision:
            cmd.extend(["--revision", variant.download.revision])
    else:  # mlx — download into HF cache
        cmd = ["hf", "download", variant.path]
        if variant.download and variant.download.revision:
            cmd.extend(["--revision", variant.download.revision])
    click.echo(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def status_line(variant: Variant) -> str:
    mark = "✓" if variant.exists_locally() else "✗"
    size = f"~{variant.approx_size_gb}GB" if variant.approx_size_gb else "?"
    return f"{mark} {variant.key:30} ({variant.fmt}, {variant.quant}, {size})"


@click.command()
@click.option("--variant", multiple=True, help="Variant key (repeatable)")
@click.option("--model", "model_id", help="Logical model id (downloads all variants)")
@click.option("--all-missing", is_flag=True, help="Download every missing variant")
@click.option("--check", is_flag=True, help="Status only — no downloads")
def main(variant: tuple, model_id: str | None, all_missing: bool, check: bool):
    registry = get_registry()

    # Filter target variants
    if variant:
        targets = [registry.variant(k) for k in variant]
    elif model_id:
        targets = registry.variants_by_model(model_id)
    elif all_missing:
        targets = [v for v in registry.variants if not v.exists_locally()]
    elif check:
        targets = registry.variants
    else:
        click.echo("Specify one of: --variant, --model, --all-missing, --check")
        sys.exit(2)

    click.echo(f"=== {len(registry.variants)} variants in registry "
               f"({sum(1 for v in registry.variants if v.exists_locally())} present) ===")
    for v in registry.variants:
        click.echo(f"  {status_line(v)}")

    if check:
        return

    click.echo()
    click.echo(f"=== Acting on {len(targets)} variant(s) ===")
    failed: list[Variant] = []
    for v in targets:
        if v.exists_locally():
            click.echo(f"  ✓ already present: {v.key}")
            continue
        click.echo(f"  ↓ downloading {v.key}")
        rc = hf_download(v)
        if rc != 0:
            click.echo(f"    FAILED rc={rc}", err=True)
            failed.append(v)

    if failed:
        click.echo(f"\n{len(failed)} download(s) failed: "
                   f"{[v.key for v in failed]}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
