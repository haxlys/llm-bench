"""Eval task groupings — priority sequence flattened for run_evals.py."""

from __future__ import annotations

# Priority sequence (mirrors the user's stated ordering):
#   1. lm-eval reasoning: MMLU + GSM8K + HellaSwag
#   2. Korean: KMMLU + HRM-8K (+ HAE-RAE, KoBEST as bonus)
#   3. Code: HumanEval + MBPP
#   4. Long context: LongBench
#   5. Tool use: BFCL (external — wrapper TBD)
#   6. Safety: TruthfulQA + ToxiGen

SUITES: dict[str, list[str]] = {
    "reasoning": ["mmlu", "gsm8k_cot_zeroshot", "hellaswag"],
    "korean":    ["kmmlu", "hrm8k", "haerae", "kobest"],
    "code":      ["humaneval_instruct", "mbpp_instruct"],
    "long":      ["longbench"],
    "safety":    ["truthfulqa", "toxigen"],
}

# A reduced version that runs in <2 min per dimension at limit=3.
SMOKE_TASKS: dict[str, list[str]] = {
    "reasoning": ["mmlu", "gsm8k_cot_zeroshot", "hellaswag"],
    "korean":    ["kmmlu", "hrm8k"],
    "code":      ["humaneval_instruct"],
    "long":      ["longbench"],          # may be heavy even at limit=3
    "safety":    ["truthfulqa"],
}


def smoke_suite() -> list[tuple[str, str]]:
    """Returns [(dim, task), ...] for smoke run."""
    return [(d, t) for d, ts in SMOKE_TASKS.items() for t in ts]


def full_suite() -> list[tuple[str, str]]:
    return [(d, t) for d, ts in SUITES.items() for t in ts]
