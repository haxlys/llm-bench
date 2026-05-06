#!/usr/bin/env python3
"""Export website data and keep public download artifacts in sync."""

from __future__ import annotations

import shutil
from pathlib import Path

from llm_bench.site_data import write_site_data

SITE_JSON = Path("site/src/data/benchmarks.json")
PUBLIC_DATA_DIR = Path("site/public/data")
PUBLIC_JSON = PUBLIC_DATA_DIR / "benchmarks.json"
STATIC_DOWNLOADS = (
    Path("results/summary.csv"),
    Path("results/eval_summary_primary.csv"),
    Path("results/mtplx_speedups.csv"),
)


def sync_site_public_data(repo_root: Path, generated_at: str | None = None) -> list[Path]:
    root = Path(repo_root)
    src_json = root / SITE_JSON
    public_dir = root / PUBLIC_DATA_DIR
    public_json = root / PUBLIC_JSON

    write_site_data(root, src_json, generated_at=generated_at)
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_json, public_json)

    copied = [src_json, public_json]
    for relative_source in STATIC_DOWNLOADS:
        source = root / relative_source
        destination = public_dir / source.name
        shutil.copyfile(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    for path in sync_site_public_data(Path.cwd()):
        print(path)


if __name__ == "__main__":
    main()
