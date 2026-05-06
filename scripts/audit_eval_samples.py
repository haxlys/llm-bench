"""Audit lm-eval sample JSONL files for common scoring failure modes.

This is intentionally heuristic. It helps distinguish "model could not answer"
from "answer exists but the task extractor did not accept the format".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "results" / "eval_scores"
OUT_DIR = ROOT / "results" / "analysis"


@dataclass
class SampleIssue:
    kind: str
    doc_id: str
    target: str
    response: str
    sample_file: str


def main() -> None:
    args = parse_args()
    variants = args.variant or [
        "gemma-4-E4B-gguf-q8",
        "qwen-3.5-4b-gguf-q8",
        "qwen-3.5-9b-gguf-q8",
        "qwen-3.6-35b-a3b-gguf-q4",
        "qwen-3-coder-next-gguf-q4",
    ]
    tasks = args.task or ["gsm8k_cot_zeroshot", "hrm8k", "leaderboard_ifeval"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    issues: list[SampleIssue] = []
    for variant in variants:
        for task in tasks:
            run_dir = latest_run_dir(variant, task)
            if run_dir is None:
                continue
            for sample_file in sorted((run_dir / task).rglob("samples_*.jsonl")):
                stats, sample_issues = audit_file(sample_file)
                rows.append({
                    "variant": variant,
                    "task": task,
                    "run_id": run_dir.name,
                    "sample_file": str(sample_file.relative_to(ROOT)),
                    **stats,
                })
                issues.extend(sample_issues)

    payload = {
        "rows": rows,
        "issues": [issue.__dict__ for issue in issues],
    }
    json_path = OUT_DIR / args.json_name
    md_path = OUT_DIR / args.markdown_name
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(rows, issues))
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {md_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--json-name", default="eval_sample_audit.json")
    parser.add_argument("--markdown-name", default="eval_sample_audit.md")
    return parser.parse_args()


def latest_run_dir(variant: str, task: str) -> Path | None:
    matches = []
    for run_dir in EVAL_DIR.glob(f"*_{variant}_full"):
        if (run_dir / task).is_dir():
            matches.append(run_dir)
    return sorted(matches)[-1] if matches else None


def audit_file(path: Path) -> tuple[dict, list[SampleIssue]]:
    n = 0
    exact_values: list[float] = []
    response_lengths: list[int] = []
    counters: defaultdict[str, int] = defaultdict(int)
    issues: list[SampleIssue] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            counters["bad_json"] += 1
            continue
        n += 1
        response = first_response(sample)
        target = str(sample.get("target", ""))
        exact = numeric(sample.get("exact_match"))
        if exact is not None:
            exact_values.append(exact)
        response_lengths.append(len(response))

        if not response.strip():
            counters["empty_response"] += 1
            issues.append(issue("empty_response", sample, target, response, path))
        if any(str(x).strip() == "[invalid]" for x in sample.get("filtered_resps", [])):
            counters["invalid_filtered_response"] += 1
        if target and target_appears_in_response(target, response):
            counters["target_appears_in_response"] += 1
            if exact == 0:
                counters["possible_extraction_miss"] += 1
                issues.append(issue("possible_extraction_miss", sample, target, response, path))
        if len(response) > 3000:
            counters["very_long_response"] += 1
            issues.append(issue("very_long_response", sample, target, response, path))

    stats = {
        "n": n,
        "exact_mean": round(mean(exact_values), 4) if exact_values else None,
        "empty_response": counters["empty_response"],
        "invalid_filtered_response": counters["invalid_filtered_response"],
        "target_appears_in_response": counters["target_appears_in_response"],
        "possible_extraction_miss": counters["possible_extraction_miss"],
        "very_long_response": counters["very_long_response"],
        "avg_response_chars": round(mean(response_lengths), 1) if response_lengths else 0.0,
    }
    return stats, issues


def first_response(sample: dict) -> str:
    resps = sample.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return ""


def numeric(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def target_appears_in_response(target: str, response: str) -> bool:
    target_norm = normalize_text(target)
    response_norm = normalize_text(response)
    if target_norm and target_norm in response_norm:
        return True
    target_nums = extract_numbers(target)
    response_nums = extract_numbers(response)
    return bool(target_nums and target_nums[-1] in response_nums)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def extract_numbers(value: str) -> list[str]:
    return [n.strip("0") or "0" for n in re.findall(r"-?\d+(?:\.\d+)?", value)]


def issue(kind: str, sample: dict, target: str, response: str, path: Path) -> SampleIssue:
    return SampleIssue(
        kind=kind,
        doc_id=str(sample.get("doc_id", "")),
        target=target[:200],
        response=response[:500],
        sample_file=str(path.relative_to(ROOT)),
    )


def render_markdown(rows: list[dict], issues: list[SampleIssue]) -> str:
    lines = [
        "# Eval Sample Audit",
        "",
        "Heuristic audit for empty responses, invalid filtered responses, long outputs, "
        "and cases where the target appears in the raw response but exact_match is zero.",
        "",
        "## Summary",
        "",
        "| Variant | Task | File | N | Exact | Empty | Invalid | Target Seen | Extract Miss? | Long | Avg Chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {task} | {file} | {n} | {exact} | {empty} | {invalid} | "
            "{seen} | {miss} | {long} | {chars} |".format(
                variant=row["variant"],
                task=row["task"],
                file=Path(row["sample_file"]).name,
                n=row["n"],
                exact="" if row["exact_mean"] is None else row["exact_mean"],
                empty=row["empty_response"],
                invalid=row["invalid_filtered_response"],
                seen=row["target_appears_in_response"],
                miss=row["possible_extraction_miss"],
                long=row["very_long_response"],
                chars=row["avg_response_chars"],
            )
        )

    grouped: defaultdict[str, list[SampleIssue]] = defaultdict(list)
    for item in issues:
        grouped[item.kind].append(item)
    lines.extend(["", "## Notable Examples", ""])
    for kind, items in sorted(grouped.items()):
        lines.extend([f"### {kind}", ""])
        for item in items[:10]:
            lines.extend([
                f"- `{item.sample_file}` doc `{item.doc_id}` target `{item.target}`",
                "",
                "```text",
                item.response,
                "```",
                "",
            ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
