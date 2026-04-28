"""Wrapper around lm-eval-harness CLI for OpenAI-compatible servers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Tasks that execute model-generated code; need explicit opt-in.
CODE_EVAL_TASKS = {"humaneval", "humaneval_instruct", "mbpp", "mbpp_instruct", "mbpp_plus"}

# Hard ceiling per task. LongBench is the slow one — ~3 hours wall on a 31B
# model at no limit. 6 hours is generous but ensures overnight matrices can't
# stall on a single task that hangs (e.g. server unresponsive after OOM).
DEFAULT_TASK_TIMEOUT_S = 6 * 60 * 60

# We use local-chat-completions: lm-eval will POST to /v1/chat/completions and
# apply the model's own chat template (set by mlx-lm-server / llama-server with --jinja).
# For tasks that prefer raw completions (e.g. some MCQ logprob tasks), set USE_CHAT=False.


def run_lmeval(
    task: str,
    base_url: str,            # e.g. http://127.0.0.1:9090/v1
    model_label: str,         # arbitrary string used as model= for client
    output_dir: Path,
    limit: int | None = None,
    batch_size: int = 1,
    num_fewshot: int | None = None,
    use_chat: bool = True,
    extra_args: list[str] | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
) -> dict:
    """Run a single lm-eval task. Returns parsed results dict (or {} on parse failure).

    Raises subprocess.TimeoutExpired if the task exceeds timeout_s; callers in
    run_evals.py catch this per-variant so other tasks still run.
    """
    if not shutil.which("lm_eval") and not _module_exists("lm_eval"):
        raise RuntimeError("lm-eval not installed (uv sync --extra evals)")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_class = "local-chat-completions" if use_chat else "local-completions"
    base_url_full = (
        base_url.rstrip("/") + "/chat/completions" if use_chat
        else base_url.rstrip("/") + "/completions"
    )
    # For chat tasks: skip tokenizer (server applies chat template internally).
    # For loglikelihood tasks: need a tokenizer to compute context lengths;
    # use huggingface backend with the model id as tokenizer source.
    if use_chat:
        tok_args = "tokenizer_backend=None,"
    else:
        tok_args = f"tokenizer_backend=huggingface,tokenizer={model_label},"
    model_args = (
        f"base_url={base_url_full},"
        f"model={model_label},"
        f"{tok_args}"
        "num_concurrent=1,"
        "max_retries=2"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_class,
        "--model_args", model_args,
        "--tasks", task,
        "--output_path", str(output_dir),
        "--batch_size", str(batch_size),
    ]
    if use_chat:
        cmd.append("--apply_chat_template")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if task in CODE_EVAL_TASKS:
        cmd.append("--confirm_run_unsafe_code")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if task in CODE_EVAL_TASKS:
        env["HF_ALLOW_CODE_EVAL"] = "1"

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        log_path = output_dir / f"{task}.log"
        log_path.write_text(
            "=== cmd ===\n" + " ".join(cmd) +
            f"\n=== TIMEOUT after {timeout_s}s ===\n" +
            (e.stdout or b"").decode("utf-8", errors="replace") +
            "\n=== stderr ===\n" +
            (e.stderr or b"").decode("utf-8", errors="replace")
        )
        return {"task": task, "error": f"timeout after {timeout_s}s", "log": str(log_path)}
    log_path = output_dir / f"{task}.log"
    log_path.write_text(
        "=== cmd ===\n" + " ".join(cmd) +
        "\n=== stdout ===\n" + proc.stdout +
        "\n=== stderr ===\n" + proc.stderr
    )

    if proc.returncode != 0:
        return {"task": task, "error": f"rc={proc.returncode}", "log": str(log_path)}

    # lm-eval writes results_*.json into output_dir/<model_args_hash>/
    return _find_latest_results(output_dir, task) or {"task": task, "error": "no results json"}


def _find_latest_results(output_dir: Path, task: str) -> dict | None:
    candidates = sorted(output_dir.rglob("results_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        # Result file may aggregate multiple tasks; we just return whatever is there.
        if "results" in data:
            return {"task": task, "results_file": str(p), "results": data["results"]}
    return None


def _module_exists(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True
