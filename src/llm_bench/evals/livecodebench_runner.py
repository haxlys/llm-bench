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
import subprocess
import sys
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RELEASE = "release_v6"
DEFAULT_TASK_TIMEOUT_S = 3 * 60 * 60
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini-2024-07-18"


def run_livecodebench(
    release: str,
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Generate + evaluate a LiveCodeBench release against an OpenAI-compatible server.

    Returns a dict with keys: task, results_file, results (pass@1) — or
    {error: ...} on failure. Mirrors run_lmeval's return shape.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "livecodebench.log"

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
    if start_date := os.environ.get("LIVE_CODE_BENCH_START_DATE"):
        cmd.extend(["--start_date", start_date])
    if end_date := os.environ.get("LIVE_CODE_BENCH_END_DATE"):
        cmd.extend(["--end_date", end_date])
    if max_tokens := os.environ.get("LIVE_CODE_BENCH_MAX_TOKENS"):
        cmd.extend(["--max_tokens", max_tokens])
    if os.environ.get("LIVE_CODE_BENCH_NOT_FAST") == "1":
        cmd.append("--not_fast")
    if limit is not None:
        cmd.append("--debug")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            cwd=str(source_checkout) if source_checkout else None,
        )
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== cmd ===\n" + " ".join(cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace") +
            "\n=== stderr ===\n" +
            (e.stderr or b"").decode("utf-8", errors="replace")
        )
        return {"task": "livecodebench",
                "error": f"timeout after {timeout_s}s",
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

    score_roots = [output_dir]
    if source_checkout:
        score_roots.append(source_checkout / "output")
    pass_at_1 = _extract_pass_at_1(score_roots, proc.stdout)
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
        }
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
    path = Path(raw).expanduser()
    main = path / "lcb_runner" / "runner" / "main.py"
    return path if main.is_file() else None
