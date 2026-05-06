"""Berkeley Function-Calling Leaderboard runner via the `bfcl_eval` PyPI package.

BFCL evaluates an LLM's tool-use ability — given a natural-language request and
a list of available functions, can the model emit a syntactically- and
semantically-correct call? This is the previously-empty 'tool' dim in our
matrix.

Install:
    uv pip install bfcl-eval

Usage:
    res = run_bfcl(
        base_url="http://127.0.0.1:9090/v1",
        model_label="lmstudio-community/...",
        output_dir=Path("results/.../bfcl"),
    )
    # res = {"task": "bfcl",
    #        "results": {"bfcl": {"overall_accuracy,none": 0.42, ...}}, ...}

Categories: by default we run the AST-evaluable, server-cheap subset
    {simple, parallel, multiple, parallel_multiple}
which covers ~700 questions and runs in <30 min on M5 Max. Live and
multi-turn categories are available via `categories=` arg but cost ~2-3×
more wall time and require external tool execution.

Returns the same dict shape as run_lmeval / run_evalplus / run_livecodebench
so aggregate.py can treat all four runners uniformly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CATEGORIES = ["simple", "parallel", "multiple", "parallel_multiple"]
DEFAULT_TASK_TIMEOUT_S = 2 * 60 * 60


def run_bfcl(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    categories: list[str] | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Generate + evaluate BFCL categories against an OpenAI-compatible server.

    Returns a dict with keys: task, results_file, results — or
    {error: ...} on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "bfcl.log"
    cats = categories or DEFAULT_CATEGORIES

    # bfcl_eval imports a tree-sitter chain at module load. Some published
    # versions ship code that's incompatible with newer tree-sitter versions
    # (e.g. `Language(ptr, name)` vs `Language(ptr)`). A bare top-level import
    # can succeed while the CLI still crashes — so probe the CLI entrypoint.
    if not _cli_loads("bfcl_eval"):
        return {
            "task": "bfcl",
            "error": ("bfcl_eval not installed or broken-installed "
                      "(`uv pip install bfcl-eval`; if it loads but `python "
                      "-m bfcl_eval --help` raises, the package version is "
                      "incompatible with installed tree-sitter — pin "
                      "`tree-sitter==0.21.3` and `tree-sitter-java==0.21.0`)"),
        }

    # bfcl's openai backend uses the standard env vars. Point it at our local
    # server. mlx_lm.server / llama-server ignore the API key.
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")

    cat_arg = ",".join(cats)
    common = ["--model", model_label, "--test-category", cat_arg]

    # Generate phase
    gen_cmd = [sys.executable, "-m", "bfcl_eval", "generate", *common,
               "--backend", "openai", "--result-dir", str(output_dir)]
    if limit is not None:
        gen_cmd.extend(["--num-tests", str(limit)])

    try:
        gen = subprocess.run(gen_cmd, capture_output=True, text=True,
                             env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== generate cmd ===\n" + " ".join(gen_cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace") +
            "\n=== stderr ===\n" +
            (e.stderr or b"").decode("utf-8", errors="replace")
        )
        return {"task": "bfcl", "error": f"generate timeout after {timeout_s}s",
                "log": str(log_path)}

    if gen.returncode != 0:
        log_path.write_text(
            "=== generate cmd ===\n" + " ".join(gen_cmd) +
            "\n=== stdout ===\n" + gen.stdout +
            "\n=== stderr ===\n" + gen.stderr
        )
        return {"task": "bfcl", "error": f"generate rc={gen.returncode}",
                "log": str(log_path)}

    # Evaluate phase
    eval_cmd = [sys.executable, "-m", "bfcl_eval", "evaluate", *common,
                "--result-dir", str(output_dir),
                "--score-dir", str(output_dir / "score")]
    try:
        ev = subprocess.run(eval_cmd, capture_output=True, text=True,
                            env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== evaluate cmd ===\n" + " ".join(eval_cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace")
        )
        return {"task": "bfcl", "error": f"evaluate timeout after {timeout_s}s",
                "log": str(log_path)}

    log_path.write_text(
        "=== generate cmd ===\n" + " ".join(gen_cmd) +
        "\n=== generate stdout ===\n" + gen.stdout +
        "\n=== generate stderr ===\n" + gen.stderr +
        "\n\n=== evaluate cmd ===\n" + " ".join(eval_cmd) +
        "\n=== evaluate stdout ===\n" + ev.stdout +
        "\n=== evaluate stderr ===\n" + ev.stderr
    )

    if ev.returncode != 0:
        return {"task": "bfcl", "error": f"evaluate rc={ev.returncode}",
                "log": str(log_path)}

    scores = _parse_scores(output_dir / "score", ev.stdout)
    if not scores:
        return {"task": "bfcl", "error": "no scores parsed from bfcl output",
                "log": str(log_path)}

    overall = scores.get("overall_accuracy")
    if overall is None and scores:
        overall = sum(scores.values()) / len(scores)

    # Synthetic results JSON so aggregate.py picks it up.
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    synthetic_path = output_dir / f"results_{ts}_bfcl.json"
    synthetic_path.write_text(json.dumps({
        "results": {
            "bfcl": {
                "overall_accuracy,none": overall,
                **{f"{cat}_acc,none": v for cat, v in scores.items()
                   if cat != "overall_accuracy"},
            }
        }
    }))

    return {
        "task": "bfcl",
        "results_file": str(synthetic_path),
        "results": {
            "bfcl": {"overall_accuracy,none": overall, **{f"{cat}_acc,none": v
                                                          for cat, v in scores.items()
                                                          if cat != "overall_accuracy"}}
        },
        "categories": cats,
    }


def _parse_scores(score_dir: Path, stdout: str) -> dict[str, float]:
    """Pull category → accuracy mapping from BFCL score artifacts.

    BFCL writes per-category JSON like score/<model>/simple_score.json
    containing {"accuracy": 0.x, ...}. Schema/path drifts between versions
    so we walk defensively and fall back to stdout regex.
    """
    out: dict[str, float] = {}

    if score_dir.exists():
        for p in score_dir.rglob("*_score.json"):
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            cat = p.stem.replace("_score", "")
            acc = _extract_accuracy(data)
            if acc is not None:
                out[cat] = acc

    # Some BFCL versions write a single overall CSV instead of per-cat JSON.
    if not out and score_dir.exists():
        for p in score_dir.rglob("*.csv"):
            for row in _csv_rows(p):
                cat = row.get("test_category") or row.get("category")
                acc_str = row.get("accuracy") or row.get("overall_acc")
                if cat and acc_str:
                    try:
                        out[cat] = float(acc_str)
                    except ValueError:
                        continue

    # Fallback: parse stdout — bfcl prints "Overall Accuracy: 0.42"
    if not out:
        import re
        for m in re.finditer(r"([\w\s]+?)\s*Accuracy[\s:=]+([\d.]+)", stdout):
            cat = m.group(1).strip().lower().replace(" ", "_")
            try:
                v = float(m.group(2))
                out[cat] = v if v <= 1.0 else v / 100.0
            except ValueError:
                continue
    return out


def _extract_accuracy(data: object) -> float | None:
    if isinstance(data, dict):
        for key in ("accuracy", "overall_accuracy", "overall_acc", "score"):
            v = data.get(key)
            if isinstance(v, (int, float)):
                return float(v) if v <= 1.0 else float(v) / 100.0
    return None


def _csv_rows(path: Path) -> list[dict[str, str]]:
    import csv
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _module_exists(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _cli_loads(name: str) -> bool:
    """True iff `python -m <name>` can at least import its entrypoint.

    bfcl_eval has versions where the top-level package imports cleanly but
    the __main__ module fails (tree-sitter API drift). Probing __main__
    catches that.
    """
    try:
        __import__(name)
    except ImportError:
        return False
    try:
        __import__(f"{name}.__main__")
    except (ImportError, TypeError, AttributeError):
        return False
    return True
