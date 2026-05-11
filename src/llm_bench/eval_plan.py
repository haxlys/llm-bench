"""Build an ordered follow-up plan from results/index.json coverage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


PRIMARY_GROUPS = [
    (
        "sourceqa_kmmlu_pro",
        "Fill source-grounding and Korean professional coverage first.",
        ("sourceqa", "kmmlu_pro"),
    ),
    (
        "primary_code",
        "Fill standard code coverage before optional agentic/code lanes.",
        ("humaneval", "mbpp", "livecodebench"),
    ),
    (
        "reasoning_instruction_catchup",
        "Catch remaining reasoning, Korean, and instruction-following primary gaps.",
        ("gsm8k_cot_zeroshot", "hrm8k", "leaderboard_ifeval"),
    ),
]

OPTIONAL_GROUPS = [
    (
        "optional_bigcodebench_bfcl_livebench",
        "Run optional frontier lanes separately after primary coverage.",
        ("bigcodebench_hard", "bfcl", "livebench_subset"),
    ),
    (
        "optional_programbench",
        "Import ProgramBench agent submission results after external runs complete.",
        ("programbench",),
    ),
]


@dataclass(frozen=True)
class TaskBucket:
    id: str
    title: str
    tasks: tuple[str, ...]
    variants: tuple[str, ...]
    command: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "tasks": list(self.tasks),
            "variants": list(self.variants),
            "command": self.command,
        }


def build_eval_catchup_plan(index_data: dict[str, Any]) -> dict[str, Any]:
    """Return ordered primary/optional/speed follow-ups from index coverage."""
    primary_missing = _coverage_by_task(index_data, "missing")
    optional_pending = _coverage_by_task(index_data, "optional")
    speed_incomplete = _speed_incomplete(index_data)

    primary = [
        bucket.as_dict()
        for bucket in _task_buckets(PRIMARY_GROUPS, primary_missing, strict=True)
        if bucket.variants
    ]
    speed = _speed_bucket(speed_incomplete)
    optional = [
        bucket.as_dict()
        for bucket in _task_buckets(OPTIONAL_GROUPS, optional_pending, strict=False)
        if bucket.variants
    ]
    return {
        "summary": {
            "primary_missing": sum(len(v) for v in primary_missing.values()),
            "optional_pending": sum(len(v) for v in optional_pending.values()),
            "speed_incomplete": len(speed_incomplete),
        },
        "primary": primary,
        "speed": speed,
        "optional": optional,
    }


def render_markdown_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# Eval Catch-up Plan",
        "",
        "Generated from `results/index.json`.",
        "",
        "## Summary",
        "",
        f"- Primary missing rows: {plan['summary']['primary_missing']}",
        f"- Optional pending rows: {plan['summary']['optional_pending']}",
        f"- Speed-incomplete variants: {plan['summary']['speed_incomplete']}",
        "",
    ]
    lines.extend(_render_bucket_section("Primary evals", plan["primary"]))
    lines.extend(_render_speed_section(plan["speed"]))
    lines.extend(_render_bucket_section("Optional lanes", plan["optional"]))
    return "\n".join(lines).rstrip() + "\n"


def _coverage_by_task(index_data: dict[str, Any], status: str) -> dict[str, set[str]]:
    by_task: dict[str, set[str]] = defaultdict(set)
    for variant in index_data.get("variants", []):
        if not isinstance(variant, dict):
            continue
        key = str(variant.get("key", ""))
        evals = variant.get("evals", {})
        if not isinstance(evals, dict):
            continue
        coverage = evals.get("coverage", [])
        if not isinstance(coverage, list):
            continue
        for row in coverage:
            if not isinstance(row, dict):
                continue
            if row.get("status") != status:
                continue
            task = str(row.get("task", ""))
            if task and key:
                by_task[task].add(key)
    return by_task


def _speed_incomplete(index_data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in index_data.get("variants", []):
        if not isinstance(variant, dict):
            continue
        speed = variant.get("speed", {})
        if not isinstance(speed, dict):
            continue
        measured = int(speed.get("scenarios_measured") or 0)
        total = int(speed.get("scenarios_total") or 0)
        if total and measured < total:
            out.append({
                "variant": str(variant.get("key", "")),
                "measured": measured,
                "total": total,
            })
    return sorted(out, key=lambda row: row["variant"])


def _task_buckets(
    groups: list[tuple[str, str, tuple[str, ...]]],
    task_to_variants: dict[str, set[str]],
    *,
    strict: bool,
) -> list[TaskBucket]:
    buckets: list[TaskBucket] = []
    for bucket_id, title, tasks in groups:
        selected_tasks = tuple(task for task in tasks if task in task_to_variants)
        variants = tuple(sorted({v for task in selected_tasks for v in task_to_variants[task]}))
        command = ""
        if variants and selected_tasks:
            command = (
                _programbench_command(variants)
                if selected_tasks == ("programbench",)
                else _run_evals_command(
                    variants=variants,
                    tasks=selected_tasks,
                    strict=strict,
                    include_bfcl="bfcl" in selected_tasks,
                )
            )
        buckets.append(TaskBucket(bucket_id, title, selected_tasks, variants, command))
    return buckets


def _speed_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = tuple(row["variant"] for row in rows)
    command = ""
    if variants:
        command = (
            "uv run python scripts/run_bench.py --skip-existing --variant "
            + " --variant ".join(variants)
        )
    return {
        "id": "speed_matrix_and_mtplx_speedup",
        "title": "Complete speed matrix and regenerate MTPLX speedups.",
        "variants": rows,
        "command": command,
        "followup_command": (
            "uv run python scripts/compare_mtplx.py && "
            "uv run python scripts/aggregate_evals.py && "
            "uv run python scripts/export_site_public_data.py"
        ),
    }


def _run_evals_command(
    *,
    variants: tuple[str, ...],
    tasks: tuple[str, ...],
    strict: bool,
    include_bfcl: bool,
) -> str:
    env = [
        f"VARIANTS=\"{' '.join(variants)}\"",
        f"TASKS=\"{' '.join(tasks)}\"",
        "SUITE=full",
        "LLM_BENCH_RESILIENT_IFEVAL=1",
    ]
    if strict:
        env.append("LLM_BENCH_STRICT_COVERAGE=1")
    if include_bfcl:
        env.append("LLM_BENCH_INCLUDE_BFCL=1")
    return " ".join(env) + " bash scripts/run_evals_overnight.sh"


def _programbench_command(variants: tuple[str, ...]) -> str:
    body = "\n".join(
        "uv run python scripts/import_programbench.py "
        f"--variant {variant} "
        "--source-dir /path/to/programbench/evaluated-run "
        "--tasks-dir /path/to/ProgramBench/src/programbench/data/tasks"
        for variant in variants
    )
    return (
        "# ProgramBench requires completed agent submissions/eval JSONs first.\n"
        + body
    )


def _render_bucket_section(title: str, buckets: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not buckets:
        lines.extend(["No pending rows.", ""])
        return lines
    for bucket in buckets:
        lines.extend([
            f"### {bucket['id']}",
            "",
            bucket["title"],
            "",
            f"- Tasks: {', '.join(bucket['tasks'])}",
            f"- Variants: {len(bucket['variants'])}",
            "",
            "```bash",
            bucket["command"],
            "```",
            "",
        ])
    return lines


def _render_speed_section(speed: dict[str, Any]) -> list[str]:
    lines = ["## Speed matrix", ""]
    variants = speed.get("variants", [])
    if not variants:
        lines.extend(["No speed-incomplete variants.", ""])
        return lines
    lines.append(speed["title"])
    lines.append("")
    for row in variants:
        lines.append(f"- {row['variant']}: {row['measured']}/{row['total']} scenarios")
    lines.extend([
        "",
        "```bash",
        speed["command"],
        speed["followup_command"],
        "```",
        "",
    ])
    return lines
