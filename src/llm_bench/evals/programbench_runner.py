"""ProgramBench result importer.

ProgramBench is agentic and heavyweight: models first create a whole program
submission, then ProgramBench evaluates that submission in Docker. This module
keeps llm-bench's first integration deliberately small by ingesting existing
``*.eval.json`` files and converting them to the shared synthetic
``results_*.json`` shape used by the other external runners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ALMOST_RESOLVED_THRESHOLD = 0.95
TASK_NAME = "programbench"


@dataclass(frozen=True)
class ProgramBenchInstanceScore:
    instance_id: str
    pass_rate: float
    n_passed: int
    n_tests: int
    resolved: bool
    almost_resolved: bool
    error_code: str | None
    n_branch_errors: int
    n_system_errors: int
    warnings: list[str]


def run_programbench_import(
    source_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    almost_threshold: float = DEFAULT_ALMOST_RESOLVED_THRESHOLD,
    tasks_dir: Path | None = None,
) -> dict:
    """Import ProgramBench eval JSON files into llm-bench result format.

    ``source_dir`` can be either a ProgramBench run output root containing
    ``<instance_id>/<instance_id>.eval.json`` files, or any directory with
    ``*.eval.json`` files below it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_files = _find_eval_files(source_dir)
    if limit is not None:
        eval_files = eval_files[:limit]
    if not eval_files:
        return {
            "task": TASK_NAME,
            "error": f"no ProgramBench *.eval.json files found under {source_dir}",
        }

    effective_tasks_dir = tasks_dir or _default_programbench_tasks_dir()
    samples = [
        asdict(
            score_programbench_eval_file(
                path,
                almost_threshold=almost_threshold,
                tasks_dir=effective_tasks_dir,
            )
        )
        for path in eval_files
    ]
    metrics = aggregate_programbench_scores(samples)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    results_path = output_dir / f"results_{ts}_{TASK_NAME}.json"
    payload = {
        "results": {TASK_NAME: metrics},
        "samples": samples,
        "source_dir": str(source_dir),
        "tasks_dir": str(effective_tasks_dir) if effective_tasks_dir else "",
        "almost_resolved_threshold": almost_threshold,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "task": TASK_NAME,
        "results_file": str(results_path),
        "samples_file": str(results_path),
        "results": {TASK_NAME: metrics},
        "source_dir": str(source_dir),
    }


def score_programbench_eval_file(
    path: Path,
    almost_threshold: float = DEFAULT_ALMOST_RESOLVED_THRESHOLD,
    tasks_dir: Path | None = None,
) -> ProgramBenchInstanceScore:
    data = json.loads(path.read_text())
    instance_id = _instance_id_from_eval_path(path)
    active_branches, ignored_tests = _load_scope(instance_id, tasks_dir)
    test_results = data.get("test_results", [])
    if not isinstance(test_results, list):
        test_results = []
    if active_branches is not None:
        test_results = [
            item
            for item in test_results
            if isinstance(item, dict)
            and item.get("branch", "") in active_branches
            and _full_test_name(item) not in ignored_tests
        ]
    n_tests = len(test_results)
    n_passed = sum(
        1
        for item in test_results
        if isinstance(item, dict) and item.get("status") == "passed"
    )
    pass_rate = n_passed / n_tests if n_tests else 0.0
    raw_branch_errors = data.get("test_branch_errors") or {}
    branch_errors = raw_branch_errors if isinstance(raw_branch_errors, dict) else {}
    n_system_errors = sum(
        1
        for item in test_results
        if isinstance(item, dict) and item.get("status") == "system_error"
    )
    error_code = data.get("error_code")
    if active_branches is not None:
        branch_errors = {
            branch: errors
            for branch, errors in branch_errors.items()
            if branch in active_branches
        }
    n_branch_errors = sum(len(errors) for errors in branch_errors.values())
    has_eval_error = bool(error_code) or bool(n_branch_errors) or n_tests == 0
    resolved = not has_eval_error and n_passed == n_tests
    almost_resolved = not has_eval_error and pass_rate >= almost_threshold
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []

    return ProgramBenchInstanceScore(
        instance_id=instance_id,
        pass_rate=round(pass_rate, 4),
        n_passed=n_passed,
        n_tests=n_tests,
        resolved=resolved,
        almost_resolved=almost_resolved,
        error_code=str(error_code) if error_code else None,
        n_branch_errors=n_branch_errors,
        n_system_errors=n_system_errors,
        warnings=[str(w) for w in warnings],
    )


def aggregate_programbench_scores(samples: list[dict]) -> dict:
    if not samples:
        return {
            "resolved_rate,none": 0.0,
            "almost_resolved_rate,none": 0.0,
            "avg_test_pass_rate,none": 0.0,
            "n_instances,none": 0,
            "n_resolved,none": 0,
            "n_almost_resolved,none": 0,
            "n_tests,none": 0,
            "n_passed_tests,none": 0,
            "n_system_error_instances,none": 0,
        }

    n = len(samples)
    n_resolved = sum(1 for sample in samples if sample["resolved"])
    n_almost = sum(1 for sample in samples if sample["almost_resolved"])
    n_tests = sum(int(sample["n_tests"]) for sample in samples)
    n_passed = sum(int(sample["n_passed"]) for sample in samples)
    n_system_error_instances = sum(
        1
        for sample in samples
        if sample["error_code"] or sample["n_branch_errors"] or sample["n_system_errors"]
    )
    return {
        "resolved_rate,none": round(n_resolved / n, 4),
        "almost_resolved_rate,none": round(n_almost / n, 4),
        "avg_test_pass_rate,none": round(
            sum(float(sample["pass_rate"]) for sample in samples) / n,
            4,
        ),
        "n_instances,none": n,
        "n_resolved,none": n_resolved,
        "n_almost_resolved,none": n_almost,
        "n_tests,none": n_tests,
        "n_passed_tests,none": n_passed,
        "n_system_error_instances,none": n_system_error_instances,
    }


def _find_eval_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.eval.json") if path.is_file())


def _instance_id_from_eval_path(path: Path) -> str:
    if path.parent.name and path.stem.startswith(path.parent.name):
        return path.parent.name
    return path.stem.removesuffix(".eval")


def _default_programbench_tasks_dir() -> Path | None:
    try:
        from programbench.constants import TASKS_DIR  # type: ignore[import-not-found]
    except ImportError:
        return None
    return Path(TASKS_DIR)


def _load_scope(
    instance_id: str,
    tasks_dir: Path | None,
) -> tuple[set[str] | None, set[str]]:
    if tasks_dir is None:
        return None, set()
    tests_json = tasks_dir / instance_id / "tests.json"
    if not tests_json.is_file():
        return None, set()
    data = json.loads(tests_json.read_text())
    branches = data.get("branches") or {}
    active_branches = {
        str(branch)
        for branch, info in branches.items()
        if isinstance(info, dict) and not info.get("ignored")
    }
    ignored_tests = {
        f"{branch}/{test['name']}"
        for branch, info in branches.items()
        if isinstance(info, dict)
        for test in info.get("ignored_tests") or []
        if isinstance(test, dict) and test.get("name")
    }
    return active_branches, ignored_tests


def _full_test_name(item: dict) -> str:
    branch = str(item.get("branch", ""))
    name = str(item.get("name", ""))
    return f"{branch}/{name}" if branch else name
