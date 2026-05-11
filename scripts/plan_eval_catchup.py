"""Write ordered benchmark follow-up plans from results/index.json."""

from __future__ import annotations

import json
from pathlib import Path

import click

from llm_bench.eval_plan import build_eval_catchup_plan, render_markdown_plan

ROOT = Path(__file__).resolve().parent.parent


@click.command()
@click.option(
    "--index",
    "index_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / "results" / "index.json",
    show_default=True,
)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=ROOT / "results" / "eval_catchup_plan.json",
    show_default=True,
)
@click.option(
    "--md-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=ROOT / "results" / "eval_catchup_plan.md",
    show_default=True,
)
def main(index_path: Path, json_out: Path, md_out: Path) -> None:
    index_data = json.loads(index_path.read_text())
    plan = build_eval_catchup_plan(index_data)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown_plan(plan))

    click.echo(f"→ wrote {json_out}")
    click.echo(f"→ wrote {md_out}")


if __name__ == "__main__":
    main()
