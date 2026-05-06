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

# Default max_gen_toks for chat tasks. mlx_lm.server has a known bug where
# finish_reason="length" (max_tokens hit before EOS) causes the response body
# to come back with content=="" — completion_tokens > 0, but the content is
# stripped because the chat-template terminator never appeared. Whatever the
# task YAML asks for, we floor max_gen_toks at this value so most generations
# finish with finish_reason="stop". Confirmed via reproducer 2026-04-29.
MIN_MAX_GEN_TOKS = 8192
LONGBENCH_MAX_GEN_TOKS = 1024
REASONING_MAX_GEN_TOKS = 1024
INSTRUCTION_MAX_GEN_TOKS = 4096

# Per-task overrides applied on top of the lm-eval-harness defaults.
# Why: the bundled task YAMLs ship choices that don't fit our setup:
#   - mbpp_instruct / humaneval_instruct: max_gen_toks=256 is too short for
#     full function bodies, so extract_code can't find a complete solution
#     and pass@1 collapses to 0. Bump high so generations finish.
#   - longbench: 21 sub-tasks with 8k-32k token prompts. Per-sample wall
#     ranges from ~5s (5k-token prefill) to ~17s (22k-token prefill) on
#     M5 Max + 26B-MoE-mlx-4bit. Empirical sample-time math:
#         limit=1   → 21 samples ≈ 4 min  (sanity only, 1 sample each)
#         limit=10  → 210 samples ≈ 45 min   (variance still high)
#         limit=50  → 1050 samples ≈ 4 h    ← chosen here
#         limit=100 → 2100 samples ≈ 8 h    (overkill for matrix)
#         limit=200 → 4200 samples ≈ 16 h   (was previous cap; too long)
#         no-limit  → 3500 samples ≈ 12 h
#     A 6-variant matrix at limit=50 = ~24h *just* on longbench, comfortably
#     under the per-task 6h timeout. stderr at n=50 per sub-task is ~7%
#     worst-case (p=0.5) — coarse, but matrix comparisons typically see
#     differences ≥ 10pp between variants so the ranking still holds.
TASK_OVERRIDES: dict[str, dict] = {
    "gsm8k_cot_zeroshot": {"gen_kwargs": f"max_gen_toks={REASONING_MAX_GEN_TOKS}"},
    "mbpp_instruct":      {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "humaneval_instruct": {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "mbpp":               {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "humaneval":          {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "mbpp_plus":          {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "mbpp_plus_instruct": {"gen_kwargs": f"max_gen_toks={MIN_MAX_GEN_TOKS}"},
    "longbench":          {"limit_cap": 50, "gen_kwargs": f"max_gen_toks={LONGBENCH_MAX_GEN_TOKS}"},
    # hrm8k has 5 sub-tasks (gsm8k, math, omni_math, mmmlu, ksm) so a CLI
    # --limit=N translates to 5*N total samples. Keep a moderate generation
    # cap here; otherwise local GGUF runs can spend >90 min on limit=20 for a
    # single 4B-class model when answers fail to stop naturally.
    "hrm8k":              {"limit_cap": 60, "gen_kwargs": f"max_gen_toks={REASONING_MAX_GEN_TOKS}"},
    # Frontier evals (HF Open LLM Leaderboard v2 task IDs):
    # MMLU-Pro: 12k Q × 10 logprob calls per sample. limit=1000 keeps gguf
    # variant wall at ~30-60 min (stderr ~1.5pp at p=0.5, sufficient for
    # variant ranking). Full set is ~6-12h per variant — too costly for a
    # 6-variant matrix.
    "leaderboard_mmlu_pro":      {"limit_cap": 1000},
    # GPQA-Diamond: 198 Q only, no cap needed.
    # IFEval: outputs should fit well below the global MLX safety default in
    # local matrix runs; 4096 avoids pathological non-stop generations.
    "leaderboard_ifeval":        {"gen_kwargs": f"max_gen_toks={INSTRUCTION_MAX_GEN_TOKS}"},
}

# We use local-chat-completions: lm-eval will POST to /v1/chat/completions and
# apply the model's own chat template (set by mlx-lm-server / llama-server with --jinja).
# For tasks that prefer raw completions (e.g. some MCQ logprob tasks), set USE_CHAT=False.


def run_lmeval(
    task: str,
    base_url: str,            # e.g. http://127.0.0.1:9090/v1
    model_label: str,         # arbitrary string used as model= for client
    output_dir: Path,
    tokenizer_label: str | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    num_fewshot: int | None = None,
    use_chat: bool = True,
    extra_args: list[str] | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
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
        tokenizer = tokenizer_label or model_label
        tok_args = f"tokenizer_backend=huggingface,tokenizer={tokenizer},"
    model_args = (
        f"base_url={base_url_full},"
        f"model={model_label},"
        f"{tok_args}"
        "num_concurrent=1,"
        "max_retries=2"
    )

    overrides = TASK_OVERRIDES.get(task, {})
    effective_limit = limit
    if (cap := overrides.get("limit_cap")) is not None:
        effective_limit = cap if effective_limit is None else min(effective_limit, cap)
    # Chat tasks without an explicit override still need a high max_gen_toks
    # to dodge the mlx_lm.server finish_reason=length empty-content bug.
    gen_kwargs_override = overrides.get("gen_kwargs")
    if gen_kwargs_override is None and use_chat:
        gen_kwargs_override = f"max_gen_toks={MIN_MAX_GEN_TOKS}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_class,
        "--model_args", model_args,
        "--tasks", task,
        "--output_path", str(output_dir),
        "--batch_size", str(batch_size),
        "--log_samples",
    ]
    if use_chat:
        cmd.append("--apply_chat_template")
    if effective_limit is not None:
        cmd.extend(["--limit", str(effective_limit)])
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if gen_kwargs_override is not None:
        cmd.extend(["--gen_kwargs", gen_kwargs_override])
    if task in CODE_EVAL_TASKS:
        cmd.append("--confirm_run_unsafe_code")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY", "local-no-auth")
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
