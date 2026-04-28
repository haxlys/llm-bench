"""Eval task groupings — priority sequence flattened for run_evals.py."""

from __future__ import annotations

# Priority sequence (mirrors the user's stated ordering):
#   1. lm-eval reasoning: MMLU + GSM8K + HellaSwag
#   2. Korean: KMMLU + HRM-8K (+ HAE-RAE, KoBEST as bonus)
#   3. Code: HumanEval + MBPP
#   4. Long context: LongBench
#   5. Tool use: BFCL (external — wrapper TBD)
#   6. Safety: TruthfulQA + ToxiGen

# mlx_lm.server's /v1/completions endpoint does not return logprobs, so
# loglikelihood-based MCQ tasks (mmlu, hellaswag, kobest, ...) cannot run on
# the MLX path. We use generative variants ('mmlu_generative', 'kmmlu_direct',
# 'truthfulqa-multi_gen_en') that ask the model to *output* the answer letter.
# Tasks that have no generative variant (hellaswag, haerae, kobest, toxigen)
# are gguf-only — they're listed here but skipped automatically when fmt=mlx.
SUITES: dict[str, list[str]] = {
    "reasoning": ["mmlu_generative", "gsm8k_cot_zeroshot", "hellaswag"],
    "korean":    ["kmmlu_direct", "hrm8k", "haerae", "kobest"],
    "code":      ["humaneval_instruct", "mbpp_instruct"],
    "long":      ["longbench"],
    "safety":    ["truthfulqa-multi_gen_en", "toxigen"],
}

# A reduced version that runs in <2 min per dimension at limit=2.
SMOKE_TASKS: dict[str, list[str]] = {
    "reasoning": ["mmlu_generative", "gsm8k_cot_zeroshot"],
    "korean":    ["kmmlu_direct", "hrm8k"],
    "code":      ["humaneval_instruct"],
    "long":      ["longbench"],          # may be heavy even at limit=3
    "safety":    ["truthfulqa-multi_gen_en"],
}


# Tasks that score via loglikelihood / multi-choice. They require the inference
# server to return per-token logprobs in /v1/completions response.
# llama-server (GGUF) supports this; mlx_lm.server does NOT — so when fmt=mlx
# these tasks are skipped entirely (see run_evals.py).
LOGLIKELIHOOD_TASKS: set[str] = {
    "mmlu", "mmlu_pro", "hellaswag", "truthfulqa", "kobest", "haerae",
    "kmmlu", "toxigen", "bbh",
}


def is_chat_task(task: str) -> bool:
    """Tasks that work via /v1/chat/completions (generative output, no logprobs needed)."""
    return task not in LOGLIKELIHOOD_TASKS


def supports_fmt(task: str, fmt: str) -> bool:
    """False if the task requires logprobs and the fmt's server doesn't supply them."""
    if fmt == "mlx" and task in LOGLIKELIHOOD_TASKS:
        return False
    return True


def smoke_suite() -> list[tuple[str, str]]:
    """Returns [(dim, task), ...] for smoke run."""
    return [(d, t) for d, ts in SMOKE_TASKS.items() for t in ts]


def full_suite() -> list[tuple[str, str]]:
    return [(d, t) for d, ts in SUITES.items() for t in ts]
