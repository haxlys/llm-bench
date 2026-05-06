"""Eval task groupings — priority sequence flattened for run_evals.py."""

from __future__ import annotations

# Priority sequence (mirrors the user's stated ordering):
#   1. lm-eval reasoning: MMLU + GSM8K + HellaSwag + MMLU-Pro + GPQA-Diamond
#   2. Korean: KMMLU + HRM-8K (+ HAE-RAE, KoBEST as bonus)
#   3. Code: HumanEval + MBPP + LiveCodeBench
#   4. Long context: LongBench
#   5. Instruction following: IFEval
#   6. Tool use: BFCL v4
#   7. Source grounding: pinned repo QA
#   8. Safety: TruthfulQA + ToxiGen
#
# Frontier additions (2026-05-01):
#   - leaderboard_mmlu_pro / leaderboard_gpqa_diamond / leaderboard_ifeval
#     mirror HF Open LLM Leaderboard v2's normalization.
#   - livecodebench (external runner) addresses HumanEval/MBPP contamination.
#   - bfcl (external runner) fills the previously-empty tool-use dim.

# mlx_lm.server's /v1/completions endpoint does not return logprobs, so
# loglikelihood-based MCQ tasks (mmlu, hellaswag, kobest, ...) cannot run on
# the MLX path. We use generative variants ('mmlu_generative', 'kmmlu_direct',
# 'truthfulqa-multi_gen_en') that ask the model to *output* the answer letter.
# Tasks that have no generative variant (hellaswag, haerae, kobest, toxigen,
# leaderboard_mmlu_pro, leaderboard_gpqa_diamond) are gguf-only — they're
# listed here but skipped automatically when fmt=mlx.
SUITES: dict[str, list[str]] = {
    # Tasks routed through lm-eval-harness only. Tasks moved to dedicated
    # runners (and removed from this list) on 2026-04-29:
    #   - mmlu_generative          → simple-evals (lm-eval until=['\n'] eats chat preamble)
    #   - kmmlu_direct             → KMMLU-Pro (lm-eval Korean YAML has same chat-mode bug)
    #   - humaneval_instruct       → EvalPlus    (lm-eval extract_code regex returns '')
    #   - mbpp_instruct            → EvalPlus    (same regex bug)
    #   - truthfulqa-multi_gen_en  → dropped     (BLEU-max under chat is noisy and unused
    #                                             in industry reports; cost > signal)
    "reasoning":   ["gsm8k_cot_zeroshot", "hellaswag",
                    "leaderboard_mmlu_pro", "leaderboard_gpqa_diamond"],
    "korean":      ["hrm8k", "haerae", "kobest"],
    "code":        [],
    "instruction": ["leaderboard_ifeval"],
    # "long": ["longbench"] — disabled 2026-04-29: a single longbench run takes
    # ~4h per variant even capped at limit=50, which dominates the matrix budget
    # without adding ranking signal that gsm8k/hrm8k don't already provide.
    # Re-enable when long-context comparison is the explicit goal of the run.
    "safety":      ["toxigen"],
}

# A reduced version that runs in <2 min per dimension at limit=2.
SMOKE_TASKS: dict[str, list[str]] = {
    "reasoning":   ["mmlu_generative", "gsm8k_cot_zeroshot"],
    "korean":      ["kmmlu_direct", "hrm8k"],
    "instruction": ["leaderboard_ifeval"],
    "safety":      ["truthfulqa-multi_gen_en"],
}

LONG_TASKS: dict[str, list[str]] = {
    "long": ["longbench"],
}


# External-runner tasks: not driven through lm-eval-harness. Each entry maps
# (dim, task_id) → runner name; run_evals.py dispatches to the right wrapper.
# Why: see SUITES comment block above. EvalPlus owns code, simple-evals owns
# Western multiple-choice generative, KMMLU-Pro owns Korean MCQ.
#
# IMPORTANT: every external runner here talks to the same OpenAI-compatible
# server we boot for the variant. mlx_lm.server has a bug where chat
# completions whose finish_reason is "length" come back with content="" (the
# llama-server / GGUF path does NOT have this bug — confirmed via curl
# 2026-04-29). Until that's fixed upstream, the runners below are
# de-facto gguf-only — they'll produce 0.0 scores against mlx variants.
EXTERNAL_SUITES: dict[str, list[tuple[str, str]]] = {
    # (dim, task_id, runner)
    "code": [
        ("humaneval", "evalplus"),
        ("mbpp", "evalplus"),
        ("livecodebench", "livecodebench"),
    ],
    # Tool-use dim: BFCL is opt-in via --include-bfcl in run_evals.py because
    # `bfcl_eval` requires an additional manual install (`uv pip install bfcl-eval`)
    # and adds ~30 min per variant.
    "tool": [("bfcl", "bfcl")],
    # Source-grounded QA: deterministic scoring over curated evidence from
    # pinned repositories. Lightweight enough to run with the default full
    # suite and supported on both MLX and GGUF because it uses chat completion.
    "source_grounding": [("sourceqa", "sourceqa")],
    # simple-evals MMLU and KMMLU-Pro are planned but not yet wired.
    # Removed from this list so run_evals.py doesn't surface 'runner not
    # implemented' noise on every variant. Re-add once the wrappers exist.
}


def external_suite() -> list[tuple[str, str, str]]:
    """Returns [(dim, task, runner), ...] for the non-lm-eval pipeline."""
    return [(d, t, r) for d, items in EXTERNAL_SUITES.items() for t, r in items]


MLX_UNSUPPORTED_EXTERNAL_TASKS = {"humaneval", "mbpp", "livecodebench", "bfcl"}


def capabilities_for_backend(backend_or_fmt: str) -> frozenset[str]:
    """Default task capabilities for legacy fmt/backend names."""
    if backend_or_fmt == "gguf":
        return frozenset({"chat", "completions", "code_eval_chat", "tool_use_eval"})
    if backend_or_fmt == "mlx":
        return frozenset({"chat", "completions"})
    if backend_or_fmt == "openai-compatible":
        return frozenset({"chat", "completions"})
    return frozenset()


def supports_capabilities(task: str, capabilities: set[str] | frozenset[str]) -> bool:
    """False when a task needs a capability the backend does not provide."""
    if task in LOGLIKELIHOOD_TASKS and "logprobs" not in capabilities:
        return False
    return True


def external_supports_capabilities(
    task: str,
    runner: str,
    capabilities: set[str] | frozenset[str],
) -> bool:
    """False when an external runner needs capabilities the backend lacks."""
    if runner in {"evalplus", "livecodebench"}:
        return "code_eval_chat" in capabilities
    if runner == "bfcl":
        return "tool_use_eval" in capabilities
    if runner == "sourceqa":
        return "chat" in capabilities
    return True


def external_supports_fmt(task: str, runner: str, fmt: str) -> bool:
    """False for external runners known to produce misleading MLX scores.

    Unlike lm-eval tasks, these wrappers do their own extraction/scoring. The
    current mlx_lm.server chat response behavior can turn length-finished
    generations into empty content, which makes code/tool evals look like real
    0.0 scores. Skip until that server behavior is known-good for each runner.
    """
    if fmt == "mlx" and task in MLX_UNSUPPORTED_EXTERNAL_TASKS:
        return False
    return external_supports_capabilities(task, runner, capabilities_for_backend(fmt))


# Tasks that score via loglikelihood / multi-choice. They require the inference
# server to return per-token logprobs in /v1/completions response.
# The current llama-server (GGUF) response schema exposes logprobs under
# choices[].logprobs.content, while lm-eval's local-completions adapter expects
# the legacy token_logprobs/top_logprobs shape. Treat logprob evals as an
# explicit endpoint capability instead of a default local-server capability.
LOGLIKELIHOOD_TASKS: set[str] = {
    "mmlu", "leaderboard_mmlu_pro", "leaderboard_gpqa_diamond",
    "hellaswag", "truthfulqa", "kobest", "haerae",
    "kmmlu", "toxigen", "bbh",
}


def is_chat_task(task: str) -> bool:
    """Tasks that work via /v1/chat/completions (generative output, no logprobs needed)."""
    return task not in LOGLIKELIHOOD_TASKS


def supports_fmt(task: str, fmt: str) -> bool:
    """False if the task requires logprobs and the fmt's server doesn't supply them."""
    return supports_capabilities(task, capabilities_for_backend(fmt))


def smoke_suite() -> list[tuple[str, str]]:
    """Returns [(dim, task), ...] for smoke run."""
    return [(d, t) for d, ts in SMOKE_TASKS.items() for t in ts]


def long_suite() -> list[tuple[str, str]]:
    """Returns [(dim, task), ...] for explicit long-context runs."""
    return [(d, t) for d, ts in LONG_TASKS.items() for t in ts]


def full_suite() -> list[tuple[str, str]]:
    return [(d, t) for d, ts in SUITES.items() for t in ts]
