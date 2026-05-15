"""Wrapper around LiveCodeBench (`lcb_runner`) for OpenAI-compatible servers.

Why this exists: HumanEval and MBPP (even the EvalPlus variants) were
released early enough that they're in most modern model training corpora.
LiveCodeBench continuously scrapes new contest problems from LeetCode,
AtCoder, and CodeForces, so each release window post-dates a model's
training cutoff and gives a contamination-free pass@1 signal.

Install:
    uv pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git

Usage:
    res = run_livecodebench(
        release="release_v6",   # 2024-11+, post-cutoff for Gemma 4
        base_url="http://127.0.0.1:9090/v1",
        model_label="lmstudio-community/...",
        output_dir=Path("results/.../livecodebench"),
    )
    # res = {"task": "livecodebench",
    #        "results": {"livecodebench": {"pass@1,none": 0.18}}, ...}

Returns the same dict shape as run_lmeval / run_evalplus so the aggregation
pipeline treats all three runners uniformly. Also writes a synthetic
results_<ts>.json into output_dir so aggregate.py's results_*.json walk
picks the score up automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RELEASE = "release_v6"
DEFAULT_TASK_TIMEOUT_S = 5 * 60 * 60
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_OPENAI_TIMEOUT_S = 900


def run_livecodebench(
    release: str,
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    timeout_s: int | None = None,
    api_key: str | None = None,
) -> dict:
    """Generate + evaluate a LiveCodeBench release against an OpenAI-compatible server.

    Returns a dict with keys: task, results_file, results (pass@1) — or
    {error: ...} on failure. Mirrors run_lmeval's return shape.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "livecodebench.log"
    task_timeout_s = _task_timeout_s(timeout_s)

    source_checkout = _source_checkout()
    if not livecodebench_available():
        return {
            "task": "livecodebench",
            "error": ("livecodebench (lcb_runner) not installed — "
                      "install from source and set LIVE_CODE_BENCH_REPO"),
        }

    # lcb_runner reads model invocation params from env (litellm-style) when
    # the model is registered as openai-compatible. Pointing OPENAI_API_BASE
    # at the local server works for the openai/openai-chat backends.
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    env["OPENAI_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")
    work_dir = _prepare_work_dir(output_dir, source_checkout)
    if source_checkout:
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(source_checkout) if not existing_pythonpath
            else f"{source_checkout}{os.pathsep}{existing_pythonpath}"
        )

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", os.environ.get("LIVE_CODE_BENCH_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        "--scenario", "codegeneration",
        "--release_version", release,
        "--n", "1",
        "--evaluate",
    ]
    start_date = os.environ.get("LIVE_CODE_BENCH_START_DATE")
    end_date = os.environ.get("LIVE_CODE_BENCH_END_DATE")
    max_tokens = os.environ.get("LIVE_CODE_BENCH_MAX_TOKENS")
    openai_timeout = os.environ.get(
        "LIVE_CODE_BENCH_OPENAI_TIMEOUT",
        str(DEFAULT_OPENAI_TIMEOUT_S),
    )
    not_fast = os.environ.get("LIVE_CODE_BENCH_NOT_FAST") == "1"
    if start_date:
        cmd.extend(["--start_date", start_date])
    if end_date:
        cmd.extend(["--end_date", end_date])
    if max_tokens:
        cmd.extend(["--max_tokens", max_tokens])
    if openai_timeout:
        cmd.extend(["--openai_timeout", openai_timeout])
    if not_fast:
        cmd.append("--not_fast")
    if limit is not None:
        cmd.append("--debug")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=task_timeout_s,
            cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== cmd ===\n" + " ".join(cmd) +
            f"\n=== TIMEOUT after {task_timeout_s}s ===\n" +
            _decode_timeout_stream(e.stdout) +
            "\n=== stderr ===\n" +
            _decode_timeout_stream(e.stderr)
        )
        return {"task": "livecodebench",
                "error": f"timeout after {task_timeout_s}s",
                "log": str(log_path)}

    log_path.write_text(
        "=== cmd ===\n" + " ".join(cmd) +
        "\n=== stdout ===\n" + proc.stdout +
        "\n=== stderr ===\n" + proc.stderr
    )

    if proc.returncode != 0:
        return {"task": "livecodebench",
                "error": f"rc={proc.returncode}",
                "log": str(log_path)}

    pass_at_1 = _extract_pass_at_1(work_dir, proc.stdout)
    if pass_at_1 is None:
        return {"task": "livecodebench",
                "error": "no pass@1 in lcb output",
                "log": str(log_path)}

    # Synthesize a results_*.json so aggregate.py picks it up alongside
    # lm-eval results without needing a special-case path.
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    synthetic_path = output_dir / f"results_{ts}_livecodebench.json"
    synthetic_path.write_text(json.dumps({
        "results": {
            "livecodebench": {
                "pass@1,none": pass_at_1,
                "release,none": release,
            }
        },
        "metadata": {
            "release": release,
            "start_date": start_date,
            "end_date": end_date,
            "not_fast": not_fast,
            "model_label": model_label,
            "openai_timeout_s": int(openai_timeout) if openai_timeout.isdigit() else openai_timeout,
            "max_tokens": int(max_tokens) if max_tokens and max_tokens.isdigit() else max_tokens,
            "problem_count": _extract_loaded_problem_count(proc.stdout),
        },
    }))

    return {
        "task": "livecodebench",
        "results_file": str(synthetic_path),
        "results": {
            "livecodebench": {
                "pass@1,none": pass_at_1,
            }
        },
    }


def _task_timeout_s(timeout_s: int | None) -> int:
    if timeout_s is not None:
        return timeout_s
    raw = os.environ.get("LIVE_CODE_BENCH_TASK_TIMEOUT_S")
    if not raw:
        return DEFAULT_TASK_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError("LIVE_CODE_BENCH_TASK_TIMEOUT_S must be an integer") from e
    if value <= 0:
        raise ValueError("LIVE_CODE_BENCH_TASK_TIMEOUT_S must be positive")
    return value


def _extract_pass_at_1(score_roots: Path | list[Path], stdout: str) -> float | None:
    """Find pass@1 from lcb_runner output. Tries JSON files first, then stdout.

    lcb_runner typically writes Scores_*.json or scores.json containing
    {"pass@1": 0.x, ...}. Filename/schema drifts between releases so we
    scan defensively.
    """
    roots = [score_roots] if isinstance(score_roots, Path) else score_roots
    # JSON files — match anything that looks like a scores file
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.rglob("Scores*.json"))
        candidates.extend(root.rglob("*scores*.json"))
        candidates.extend(root.rglob("*_eval.json"))
    for p in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        v = _find_pass_at_1_in_dict(data)
        if v is not None:
            return v

    # Fallback: parse stdout for "pass@1: 0.123" or "pass@1 = 0.123"
    import re
    for m in re.finditer(r"pass@1[\s:=]+([\d.]+)", stdout):
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
            if 0.0 <= v <= 100.0:
                return v / 100.0
        except ValueError:
            continue
    for line in stdout.splitlines():
        raw = line.strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 0.0 <= v <= 100.0:
            return v / 100.0
    return None


def _find_pass_at_1_in_dict(d: object) -> float | None:
    """DFS for the first numeric value under a 'pass@1' or 'pass_at_1' key."""
    if isinstance(d, dict):
        for k, v in d.items():
            ks = str(k).lower()
            if ks in ("pass@1", "pass_at_1") and isinstance(v, (int, float)):
                return float(v) if v <= 1.0 else float(v) / 100.0
            inner = _find_pass_at_1_in_dict(v)
            if inner is not None:
                return inner
    elif isinstance(d, list):
        for item in d:
            inner = _find_pass_at_1_in_dict(item)
            if inner is not None:
                return inner
    return None


def _extract_loaded_problem_count(stdout: str) -> int | None:
    import re
    match = re.search(r"Loaded\s+(\d+)\s+problems", stdout)
    return int(match.group(1)) if match else None


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _prepare_work_dir(output_dir: Path, source_checkout: Path | None) -> Path:
    """Create a per-run cwd that keeps lcb_runner's relative file reads working."""
    work_dir = output_dir / "_lcb_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    package_dir = (
        source_checkout / "lcb_runner"
        if source_checkout
        else _installed_package_dir("lcb_runner")
    )
    if package_dir is None:
        return work_dir
    package_dir = package_dir.resolve()

    target = work_dir / "lcb_runner"
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == package_dir.resolve():
            return work_dir
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(package_dir, target_is_directory=True)
    return work_dir


def _installed_package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if not spec or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def livecodebench_available() -> bool:
    return _module_exists("lcb_runner.runner.main") or _source_checkout() is not None


def _source_checkout() -> Path | None:
    raw = os.environ.get("LIVE_CODE_BENCH_REPO")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    main = path / "lcb_runner" / "runner" / "main.py"
    return path if main.is_file() else None
