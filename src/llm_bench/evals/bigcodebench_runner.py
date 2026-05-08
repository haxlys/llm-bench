"""BigCodeBench-Hard wrapper for OpenAI-compatible servers."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TASK_TIMEOUT_S = 3 * 60 * 60
DEFAULT_SPLIT = "instruct"
DEFAULT_SUBSET = "hard"


def run_bigcodebench_hard(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
    execution: str | None = None,
) -> dict:
    """Generate and evaluate BigCodeBench-Hard.

    Uses upstream `bigcodebench.evaluate` with the OpenAI backend and writes a
    synthetic `results_*.json` so the shared aggregation pipeline can ingest it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "bigcodebench_hard.log"
    if not bigcodebench_available():
        return {
            "task": "bigcodebench_hard",
            "error": (
                "bigcodebench not installed; install with "
                "`uv pip install bigcodebench --upgrade`"
            ),
        }

    execution_backend = execution or os.environ.get("BIGCODEBENCH_EXECUTION", "gradio")
    cmd = [
        sys.executable,
        "-m",
        "bigcodebench.evaluate",
        "--model",
        model_label,
        "--backend",
        "openai",
        "--base_url",
        base_url.rstrip("/"),
        "--execution",
        execution_backend,
        "--split",
        DEFAULT_SPLIT,
        "--subset",
        DEFAULT_SUBSET,
        "--root",
        str(output_dir),
        "--bs",
        "1",
        "--n_samples",
        "1",
        "--temperature",
        "0.0",
        "--greedy",
        "--pass_k",
        "1",
    ]
    if limit is not None:
        cmd.extend(["--id_range", f"0-{limit}"])
    if max_new_tokens := os.environ.get("BIGCODEBENCH_MAX_NEW_TOKENS"):
        cmd.extend(["--max_new_tokens", max_new_tokens])
    if parallel := os.environ.get("BIGCODEBENCH_PARALLEL"):
        cmd.extend(["--parallel", parallel])
    if endpoint := os.environ.get("BIGCODEBENCH_GRADIO_ENDPOINT"):
        cmd.extend(["--gradio_endpoint", endpoint])

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
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
            "task": "bigcodebench_hard",
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
            "task": "bigcodebench_hard",
            "error": f"rc={proc.returncode}",
            "log": str(log_path),
        }

    pass_at_1 = _extract_pass_at_1(output_dir, proc.stdout)
    if pass_at_1 is None:
        return {
            "task": "bigcodebench_hard",
            "error": "no pass@1 parsed from BigCodeBench output",
            "log": str(log_path),
        }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    synthetic_path = output_dir / f"results_{ts}_bigcodebench_hard.json"
    synthetic_path.write_text(json.dumps({
        "results": {
            "bigcodebench_hard": {
                "pass@1,none": pass_at_1,
            }
        }
    }))
    return {
        "task": "bigcodebench_hard",
        "results_file": str(synthetic_path),
        "results": {
            "bigcodebench_hard": {
                "pass@1,none": pass_at_1,
            }
        },
        "split": DEFAULT_SPLIT,
        "subset": DEFAULT_SUBSET,
        "execution": execution_backend,
    }


def bigcodebench_available() -> bool:
    return _module_exists("bigcodebench.evaluate")


def _extract_pass_at_1(output_dir: Path, stdout: str) -> float | None:
    candidates = list(output_dir.rglob("*pass_at_k.json"))
    candidates.extend(output_dir.rglob("*pass@k*.json"))
    candidates.extend(output_dir.rglob("*eval_results.json"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        value = _find_pass_at_1(data)
        if value is not None:
            return value

    import re
    for m in re.finditer(r"pass@1[\s:=]+([\d.]+)", stdout, flags=re.IGNORECASE):
        try:
            return _normalize(float(m.group(1)))
        except ValueError:
            continue
    return None


def _find_pass_at_1(data: object) -> float | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in {"pass@1", "pass_at_1"} and isinstance(value, (int, float)):
                return _normalize(float(value))
            nested = _find_pass_at_1(value)
            if nested is not None:
                return nested
    elif isinstance(data, list):
        for item in data:
            nested = _find_pass_at_1(item)
            if nested is not None:
                return nested
    return None


def _normalize(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
