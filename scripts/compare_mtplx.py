"""Write MTPLX MTP-on/off speedup report from raw benchmark JSONs."""

from __future__ import annotations

from pathlib import Path

import click

from llm_bench.mtplx_compare import write_speedup_report

ROOT = Path(__file__).resolve().parent.parent


@click.command()
@click.option(
    "--raw-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=ROOT / "results" / "raw",
    show_default=True,
)
@click.option(
    "--out",
    "out_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=ROOT / "results" / "mtplx_speedups.csv",
    show_default=True,
)
def main(raw_dir: Path, out_csv: Path):
    out = write_speedup_report(raw_dir, out_csv)
    click.echo(f"→ MTPLX speedup report written to {out}")


if __name__ == "__main__":
    main()
