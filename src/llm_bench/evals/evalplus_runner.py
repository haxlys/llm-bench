"""Wrapper around evalplus CLI for OpenAI-compatible servers.

Why this exists: lm-eval-harness 0.4.11's `extract_code_blocks` filter (used
by mbpp_instruct / humaneval_instruct) has an open bug — when the model
emits ```python\\ncode\\n```, the prefix-and-regex extractor returns ''.
EvalPlus has a purpose-built chat-mode code path and ships HumanEval+ /
MBPP+ which use 35-80x more test cases.

Usage:
    res = run_evalplus(
        dataset="humaneval",  # or "mbpp"
        base_url="http://127.0.0.1:9090/v1",
        model_label="lmstudio-community/...",
        output_dir=Path("results/evalplus/<run_id>"),
    )
    # res = {"task": "humaneval", "results": {"pass@1": 0.61, "pass@1_plus": 0.55}, ...}

Returns the same dict shape that lm_eval's `run_lmeval` returns so the
aggregation pipeline can treat both runners uniformly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVALPLUS_TASKS = {"humaneval", "mbpp"}

# Per-dataset wall-time ceiling. EvalPlus runs all 164 (humaneval) / 378 (mbpp)
# problems sequentially against the model server; on M5 Max + 26B-MoE-mlx-4bit
# we measure ~3-5s per problem at greedy, so 2h is generous headroom.
DEFAULT_TASK_TIMEOUT_S = 2 * 60 * 60


def run_evalplus(
    dataset: str,
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Run evalplus codegen + evaluate against an OpenAI-compatible server.

    Returns a dict with keys: task, results (pass@1 / pass@1_plus), or
    {error: ...} on failure. Mirrors the shape of run_lmeval's return value.
    """
    if dataset not in EVALPLUS_TASKS:
        return {"task": dataset, "error": f"unknown evalplus dataset: {dataset}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{dataset}.log"

    # base_url here is .../v1; evalplus expects the same root.
    codegen_cmd = [
        sys.executable, "-m", "evalplus.codegen",
        "--model", model_label,
        "--dataset", dataset,
        "--backend", "openai",
        "--base_url", base_url,
        "--root", str(output_dir),
        "--greedy",
    ]
    if limit is not None:
        codegen_cmd.append(f"--id_range=[0,{limit}]")

    env = os.environ.copy()
    # evalplus's openai backend reads OPENAI_API_KEY but a local server doesn't
    # care — just give it something non-empty so the SDK doesn't refuse to call.
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")
    # EvalPlus's sandbox raises "current limit exceeds maximum limit" on macOS
    # when it tries to raise RLIMIT_AS. -1 disables that memory guard.
    env.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

    try:
        codegen = subprocess.run(
            codegen_cmd, capture_output=True, text=True,
            env=env, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== codegen cmd ===\n" + " ".join(codegen_cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace") +
            "\n=== stderr ===\n" +
            (e.stderr or b"").decode("utf-8", errors="replace")
        )
        return {"task": dataset, "error": f"codegen timeout after {timeout_s}s",
                "log": str(log_path)}

    if codegen.returncode != 0:
        log_path.write_text(
            "=== codegen cmd ===\n" + " ".join(codegen_cmd) +
            "\n=== stdout ===\n" + codegen.stdout +
            "\n=== stderr ===\n" + codegen.stderr
        )
        return {"task": dataset, "error": f"codegen rc={codegen.returncode}",
                "log": str(log_path)}

    # codegen writes evalplus_results/<dataset>/<sanitized-model>/samples.jsonl
    # but with --root override it's <output_dir>/<dataset>/<sanitized-model>/samples.jsonl
    samples_path = _find_latest_samples(output_dir)
    if not samples_path:
        log_path.write_text(
            "=== codegen cmd ===\n" + " ".join(codegen_cmd) +
            "\n=== stdout ===\n" + codegen.stdout +
            "\n=== stderr ===\n" + codegen.stderr +
            "\n=== no samples.jsonl produced ==="
        )
        return {"task": dataset, "error": "no samples.jsonl after codegen",
                "log": str(log_path)}

    eval_cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(samples_path),
    ]
    try:
        eval_proc = subprocess.run(
            eval_cmd, capture_output=True, text=True,
            env=env, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            "=== evaluate cmd ===\n" + " ".join(eval_cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace")
        )
        return {"task": dataset, "error": f"evaluate timeout after {timeout_s}s",
                "log": str(log_path)}

    log_path.write_text(
        "=== codegen cmd ===\n" + " ".join(codegen_cmd) +
        "\n=== codegen stdout ===\n" + codegen.stdout +
        "\n=== codegen stderr ===\n" + codegen.stderr +
        "\n\n=== evaluate cmd ===\n" + " ".join(eval_cmd) +
        "\n=== evaluate stdout ===\n" + eval_proc.stdout +
        "\n=== evaluate stderr ===\n" + eval_proc.stderr
    )

    if eval_proc.returncode != 0:
        return {"task": dataset, "error": f"evaluate rc={eval_proc.returncode}",
                "log": str(log_path)}

    # evaluate writes <samples_dir>/<samples-basename>_eval_results.json next to
    # the samples file with pass@1 numbers.
    eval_results = list(samples_path.parent.glob("*_eval_results.json"))
    if not eval_results:
        return {"task": dataset, "error": "no eval results json",
                "log": str(log_path)}
    results_path = max(eval_results, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(results_path.read_text())
    except json.JSONDecodeError as e:
        return {"task": dataset, "error": f"results parse error: {e}",
                "log": str(log_path)}

    # evalplus stdout contains the headline pass@1 (base) and pass@1+ (plus)
    # numbers, but we read from the JSON for stability.
    pass_at_1 = _extract_pass_at_1(data, plus=False)
    pass_at_1_plus = _extract_pass_at_1(data, plus=True)
    synthetic_path = _write_synthetic_results(
        output_dir,
        dataset,
        pass_at_1,
        pass_at_1_plus,
    )

    return {
        "task": dataset,
        "results_file": str(synthetic_path),
        "evalplus_results_file": str(results_path),
        "results": {
            dataset: {
                "pass_at_1,base": pass_at_1,
                "pass_at_1,plus": pass_at_1_plus,
            }
        },
    }


def _write_synthetic_results(
    output_dir: Path,
    dataset: str,
    pass_at_1: float | None,
    pass_at_1_plus: float | None,
) -> Path:
    """Write aggregate-compatible results_*.json for EvalPlus outputs."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = output_dir / f"results_{ts}_{dataset}.json"
    path.write_text(json.dumps({
        "results": {
            dataset: {
                "pass_at_1,base": pass_at_1,
                "pass_at_1,plus": pass_at_1_plus,
            }
        }
    }))
    return path


def _extract_pass_at_1(data: dict, plus: bool) -> float | None:
    """Pull the headline pass@1 (or pass@1+) from evalplus's eval_results.json.

    Schema: {eval: {<task_id>: [{base_status, plus_status, ...}, ...], ...},
             pass_at_k: {pass@1: 0.x, pass@1+: 0.y}}
    """
    if not isinstance(data, dict):
        return None
    pak = data.get("pass_at_k")
    if isinstance(pak, dict):
        key = "pass@1+" if plus else "pass@1"
        v = pak.get(key)
        if isinstance(v, (int, float)):
            return float(v)

    # Fallback: compute from per-task statuses if pass_at_k missing.
    eval_block = data.get("eval")
    if not isinstance(eval_block, dict) or not eval_block:
        return None
    field = "plus_status" if plus else "base_status"
    total = 0
    passed = 0
    for runs in eval_block.values():
        if not isinstance(runs, list) or not runs:
            continue
        total += 1
        if runs[0].get(field) == "pass":
            passed += 1
    return passed / total if total else None


def _find_latest_samples(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("samples.jsonl"))
    candidates.extend(
        p for p in output_dir.rglob("*.jsonl")
        if not p.name.endswith(".raw.jsonl")
    )
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
