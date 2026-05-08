"""LiveBench wrapper for OpenAI-compatible servers.

LiveBench is intentionally kept as an external checkout because its runner
expects to be executed from the upstream `livebench/` directory and shells out
to sibling scripts. Set LIVEBENCH_REPO to either the repository root or the
inner `livebench` directory.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BENCH_NAMES = [
    "live_bench/reasoning",
    "live_bench/math",
    "live_bench/instruction_following",
    "live_bench/data_analysis",
]
DEFAULT_RELEASE = "2024-11-25"
DEFAULT_TASK_TIMEOUT_S = 3 * 60 * 60


def run_livebench(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    bench_names: list[str] | None = None,
    release: str = DEFAULT_RELEASE,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Run a LiveBench subset and write aggregate-compatible results.

    Returns a dict with keys: task, results_file, results, bench_names, release;
    or `{error: ...}` when the upstream runner is unavailable or fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "livebench.log"
    livebench_dir = _livebench_dir()
    if livebench_dir is None:
        return {
            "task": "livebench_subset",
            "error": (
                "LiveBench checkout not found; set LIVEBENCH_REPO to the repo "
                "root or inner livebench directory after `pip install -e .`"
            ),
        }

    display_name = _safe_model_display_name(model_label)
    benches = bench_names or DEFAULT_BENCH_NAMES
    cmd = [
        sys.executable,
        "run_livebench.py",
        "--mode",
        "single",
        "--model",
        model_label,
        "--model-display-name",
        display_name,
        "--bench-name",
        *benches,
        "--api-base",
        base_url.rstrip("/"),
        "--api-key",
        api_key or os.environ.get("OPENAI_API_KEY", "local-no-auth"),
        "--parallel-requests",
        os.environ.get("LIVEBENCH_PARALLEL_REQUESTS", "1"),
        "--parallel-grading",
        os.environ.get("LIVEBENCH_PARALLEL_GRADING", "1"),
        "--force-temperature",
        "0.0",
        "--livebench-release-option",
        release,
    ]
    if max_tokens := os.environ.get("LIVEBENCH_MAX_TOKENS"):
        cmd.extend(["--max-tokens", max_tokens])
    if limit is not None:
        cmd.extend(["--question-begin", "0", "--question-end", str(limit)])

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")
    env["LIVEBENCH_API_KEY"] = env["OPENAI_API_KEY"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            cwd=str(livebench_dir),
        )
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== cmd ===\n"
            + " ".join(cmd)
            + f"\n=== TIMEOUT after {timeout_s}s ===\n"
            + _timeout_text(e.stdout)
            + "\n=== stderr ===\n"
            + _timeout_text(e.stderr)
        )
        return {
            "task": "livebench_subset",
            "error": f"timeout after {timeout_s}s",
            "log": str(log_path),
        }

    log_path.write_text(
        "=== cmd ===\n"
        + " ".join(cmd)
        + "\n=== stdout ===\n"
        + proc.stdout
        + "\n=== stderr ===\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        return {
            "task": "livebench_subset",
            "error": f"rc={proc.returncode}",
            "log": str(log_path),
        }

    scores = _parse_livebench_scores([livebench_dir, output_dir], proc.stdout, display_name)
    if not scores:
        return {
            "task": "livebench_subset",
            "error": "no LiveBench scores parsed",
            "log": str(log_path),
        }

    overall = scores.get("score")
    if overall is None:
        category_scores = [v for k, v in scores.items() if k.endswith("_score")]
        overall = sum(category_scores) / len(category_scores) if category_scores else None
    if overall is None:
        return {
            "task": "livebench_subset",
            "error": "no LiveBench overall score parsed",
            "log": str(log_path),
        }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    synthetic_path = output_dir / f"results_{ts}_livebench_subset.json"
    metrics = {"score,none": overall}
    metrics.update({f"{k},none": v for k, v in scores.items() if k != "score"})
    synthetic_path.write_text(json.dumps({"results": {"livebench_subset": metrics}}))

    return {
        "task": "livebench_subset",
        "results_file": str(synthetic_path),
        "results": {"livebench_subset": metrics},
        "bench_names": benches,
        "release": release,
    }


def livebench_available() -> bool:
    return _livebench_dir() is not None


def _livebench_dir() -> Path | None:
    raw = os.environ.get("LIVEBENCH_REPO")
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path, path / "livebench"]
    for candidate in candidates:
        if (candidate / "run_livebench.py").is_file():
            return candidate
    return None


def _safe_model_display_name(model_label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_label).strip("_")
    return cleaned or "local_model"


def _parse_livebench_scores(
    roots: list[Path],
    stdout: str,
    model_display_name: str,
) -> dict[str, float]:
    """Parse LiveBench result CSVs with a stdout fallback.

    Upstream writes `all_groups.csv` and `all_tasks.csv`, but column names and
    locations have shifted between versions. This parser accepts both narrow
    rows (`category, score`) and wider leaderboard-style rows.
    """
    scores: dict[str, float] = {}
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("all_groups.csv"))
            candidates.extend(root.rglob("all_tasks.csv"))
            candidates.extend(root.rglob("*livebench*.csv"))

    for path in sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True):
        for row in _csv_rows(path):
            if not _row_matches_model(row, model_display_name):
                continue
            label = _row_label(row) or path.stem
            scores.update(_row_scores(row, label))

    if scores:
        return scores

    # Fallback for lines such as "Overall: 52.1" or "score = 0.521".
    for m in re.finditer(r"(?im)\b(overall|score)\b\s*[:=]\s*([\d.]+)", stdout):
        value = _normalize_score(float(m.group(2)))
        scores["score"] = value
        break
    return scores


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _row_matches_model(row: dict[str, str], model_display_name: str) -> bool:
    model_values = [
        row.get("model"),
        row.get("model_name"),
        row.get("model_display_name"),
        row.get("Model"),
        row.get(""),
        row.get("Unnamed: 0"),
    ]
    present = [str(v) for v in model_values if v]
    return not present or model_display_name in present


def _row_label(row: dict[str, str]) -> str | None:
    for key in ("category", "group", "task", "bench_name", "name"):
        raw = row.get(key) or row.get(key.title())
        if raw:
            return _slug(str(raw).split("/")[-1])
    return None


def _row_score(row: dict[str, str]) -> float | None:
    for key in ("score", "Score", "accuracy", "Accuracy", "correct"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return _normalize_score(float(raw))
        except ValueError:
            continue
    return None


def _row_scores(row: dict[str, str], label: str) -> dict[str, float]:
    # Narrow shape: model,category,score
    score = _row_score(row)
    if score is not None:
        key = "score" if label in {"overall", "livebench", "live_bench"} else f"{label}_score"
        return {key: score}

    # Wide shape from pandas pivot: model,average,math,reasoning,...
    out: dict[str, float] = {}
    model_columns = {"", "model", "model_name", "model_display_name", "Model", "Unnamed: 0"}
    for key, raw in row.items():
        if key in model_columns or raw is None or raw == "":
            continue
        try:
            value = _normalize_score(float(raw))
        except ValueError:
            continue
        metric_key = "score" if key.lower() == "average" else f"{_slug(key)}_score"
        out[metric_key] = value
    return out


def _normalize_score(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def _timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
