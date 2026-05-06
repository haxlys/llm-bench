# Local Model Policy

Updated: 2026-05-05

This policy ranks locally downloaded GGUF models for this benchmark project.
Scores are from deterministic local evals unless noted. Higher is better.

## Default choices

| Use case | Default | Why |
|---|---|---|
| General benchmark default | `qwen-3-coder-next-gguf-q4` | Best overall reasoning/code/source profile on this machine. |
| Code-heavy evaluation | `qwen-3-coder-next-gguf-q4` | HumanEval 0.909, MBPP 0.918, LiveCodeBench window 0.250. |
| Instruction-following baseline | `gemma-4-E4B-gguf-q8` | Best IFEval strict score at 0.920, with good GSM8K 0.700. |
| Small/fast budget baseline | `qwen-3.5-4b-gguf-q8` | Useful as a lower-cost control, but not a quality leader. |
| Not recommended as default | `qwen-3.6-35b-a3b-gguf-q4` | Poor reasoning/code scores despite acceptable IFEval. |

## Current headline scores

| Variant | GSM8K | HRM8K | IFEval | SourceQA | HumanEval | MBPP | LCB window |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen-3-coder-next-gguf-q4` | 0.860 | 0.428 | 0.840 | 1.000 | 0.909 | 0.918 | 0.250 |
| `gemma-4-E4B-gguf-q8` | 0.700 | 0.312 | 0.920 | 0.900 | 0.110 | 0.320 | 0.083 |
| `qwen-3.5-4b-gguf-q8` | 0.240 | 0.088 | 0.340 | 1.000 | 0.366 | 0.677 | 0.000 |
| `qwen-3.6-35b-a3b-gguf-q4` | 0.050 | 0.040 | 0.750 | 1.000 | 0.024 | 0.085 | n/a |

Notes:

- GSM8K/HRM8K/IFEval/SourceQA are `limit=50` for the top three models, except older non-top rows are from the latest available run.
- HumanEval and MBPP are EvalPlus full runs. The table shows base pass@1; plus pass@1 is preserved in `results/eval_summary_full.csv`.
- LiveCodeBench is the 2025-03-29 to 2025-04-06 release window, 24 problems, `max_tokens=1024`. It is a practical freshness signal, not a full no-limit LCB replacement.
- `qwen-3-coder-next-gguf-q4` HRM8K default `max_gen_toks=1024` failed near 235/250 samples with a local server 500 parse error. The reported HRM8K score is the successful retry with `max_gen_toks=512`; the original failed trace is retained.
- The sample audit shows many extraction misses for `qwen-3-coder-next-gguf-q4` on HRM8K KSM/OmniMath, so its true math-answering quality may be higher than exact-match reports. Gemma and Qwen 4B show many empty responses on some HRM8K subtasks, so their low HRM rows are mostly real runner/model behavior, not just scorer noise.

## Decision rules

Use `qwen-3-coder-next-gguf-q4` when the benchmark needs one strong local model. It is the only candidate that is simultaneously strong on GSM8K, HRM8K, EvalPlus, LiveCodeBench, and SourceQA.

Use `gemma-4-E4B-gguf-q8` when instruction following is the center of the test or when a non-Qwen comparison point is valuable. Its code scores are much weaker, so do not use it as the code default.

Keep `qwen-3.5-4b-gguf-q8` as a small baseline. It is useful for speed/cost comparisons and regression tests, not as the representative quality ceiling.

Avoid promoting the 35B-A3B Qwen row until its output formatting/scoring behavior is understood. The larger size did not translate into better local deterministic benchmark scores here.

## Reproducibility policy

Primary scores should remain deterministic rule/checker scores. Optional judge-model evaluations may be added as a secondary signal, but they must not overwrite the primary score.

When a task needs a practical cap, the cap belongs in the trace or documentation. Full no-limit EvalPlus/LiveCodeBench runs should be reserved for final release gates because they are expensive and less useful for daily model selection.
