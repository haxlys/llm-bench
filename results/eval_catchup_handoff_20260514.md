# Eval Catch-up Handoff - 2026-05-14

This note summarizes the current benchmark cleanup state after the recent task
catch-up work. Use it together with `results/eval_catchup_plan.md`, which is
machine-generated from `results/index.json`.

## Current State

- No benchmark or local model server process is running.
- Aggregation and site export have been refreshed:
  - `results/eval_summary_full.csv`
  - `results/eval_summary_primary.csv`
  - `results/index.json`
  - `site/public/data/*`
  - `site/src/data/benchmarks.json`
  - `results/eval_catchup_plan.md`
  - `results/eval_catchup_plan.json`
- `sourceqa` and `kmmlu_pro` are filled for the recent model set checked in this pass.
- The main remaining gap is code coverage, especially `livecodebench`.

## Code Changes In This Worktree

- `src/llm_bench/evals/kmmlu_pro_runner.py`
  - Uses a short answer-only system prompt.
  - Caps default generation to 4 tokens.
  - Stabilizes extraction for A/B/C/D/E answers.
- `src/llm_bench/evals/server.py`
  - Passes GGUF llama-server flags to disable reasoning/thinking during evals.
- `src/llm_bench/evals/livecodebench_runner.py`
  - Runs LiveCodeBench in an isolated `_lcb_work` directory.
  - Prevents stale results under `.external/LiveCodeBench/output` from being counted.
  - Adds run metadata such as release, date window, problem count, timeout, and model label.
  - Supports `LIVE_CODE_BENCH_OPENAI_TIMEOUT`.
  - Raises the default wrapper timeout to 5 hours.
- `src/llm_bench/manifest.py`
  - Counts small but valid synthetic `results_*.json` files as measured.
- `src/llm_bench/evals/evalplus_runner.py`
  - Supports `EVALPLUS_TASK_TIMEOUT_S`.
- Tests were updated for the above behavior.

## Completed Results To Keep

- `26B-MoE-gguf-q4`
  - HumanEval done.
  - MBPP done.
  - LiveCodeBench done for `2025-01-01..2025-04-30`, 182 problems, pass@1 `0.489010989010989`.
- `26B-MoE-gguf-q8`
  - HumanEval done.
  - MBPP done.
  - LiveCodeBench done for `2025-01-01..2025-04-30`, 182 problems, pass@1 `0.4835164835164835`.
- `31B-Dense-gguf-q8`
  - HumanEval done.
  - MBPP done.
  - LiveCodeBench not completed.
- KMMLU-Pro reruns completed for the recent variants checked in this pass.

## Important LiveCodeBench Note

Use this date window for comparable new LiveCodeBench runs:

```bash
export LIVE_CODE_BENCH_REPO=.external/LiveCodeBench
export LIVE_CODE_BENCH_START_DATE=2025-01-01
export LIVE_CODE_BENCH_END_DATE=2025-04-30
export LIVE_CODE_BENCH_OPENAI_TIMEOUT=1800
export EVALPLUS_TASK_TIMEOUT_S=14400
```

The chosen window has 182 problems. Full `release_v6` has 1055 problems and is
too slow for this local catch-up pass.

Older LiveCodeBench results exist for these variants, but they have no window or
problem-count metadata and should not be treated as fully comparable to the new
182-problem window:

- `gemma-4-E4B-gguf-q8`
- `qwen-3-coder-next-gguf-q4`
- `qwen-3.5-4b-gguf-q8`

## Primary Remaining Work

Generated plan summary after refresh:

- Primary missing rows: 21
- Optional pending rows: 62
- Speed-incomplete variants: 6

Highest priority:

```bash
export LIVE_CODE_BENCH_REPO=.external/LiveCodeBench
export LIVE_CODE_BENCH_START_DATE=2025-01-01
export LIVE_CODE_BENCH_END_DATE=2025-04-30
export LIVE_CODE_BENCH_OPENAI_TIMEOUT=1800
export EVALPLUS_TASK_TIMEOUT_S=14400

VARIANTS="31B-Dense-gguf-q8 gpt-oss-120b-gguf-q4 gpt-oss-20b-gguf-q4 nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4 qwen-3-coder-30b-a3b-instruct-gguf-q4 qwen-3-next-80b-a3b-instruct-gguf-q4 qwen-3.5-9b-gguf-q8 qwen-3.6-35b-a3b-gguf-q4" \
TASKS="humaneval mbpp livecodebench" \
SUITE=full \
LLM_BENCH_RESILIENT_IFEVAL=1 \
LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh
```

This wrapper passes `--skip-existing`, so completed HumanEval/MBPP rows should
be skipped and only missing tasks should run.

Second priority:

```bash
VARIANTS="26B-MoE-gguf-q4 26B-MoE-mlx-4bit 31B-Dense-gguf-q8 31B-Dense-mlx-8bit" \
TASKS="gsm8k_cot_zeroshot hrm8k leaderboard_ifeval" \
SUITE=full \
LLM_BENCH_RESILIENT_IFEVAL=1 \
LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh
```

## Specific Known Incomplete Items

- `31B-Dense-gguf-q8`
  - `livecodebench` timed out at 3 hours after 166/182 problems.
  - A 5-hour retry was started but stopped by request, so it ended with `rc=-15`.
- `qwen-3.5-9b-gguf-q8`
  - Missing `humaneval`, `mbpp`, and `livecodebench`.
- LiveCodeBench-only missing:
  - `qwen-3.6-35b-a3b-gguf-q4`
  - `qwen-3-coder-30b-a3b-instruct-gguf-q4`
  - `qwen-3-next-80b-a3b-instruct-gguf-q4`
  - `gpt-oss-120b-gguf-q4`
  - `gpt-oss-20b-gguf-q4`
  - `nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4`

## After Running More Evals

Always refresh derived data:

```bash
uv run python scripts/aggregate_evals.py
uv run python scripts/export_site_public_data.py
uv run python scripts/plan_eval_catchup.py
```

Recommended verification for runner changes:

```bash
LIVE_CODE_BENCH_REPO=.external/LiveCodeBench uv run pytest \
  tests/test_livecodebench_runner.py \
  tests/test_evalplus_runner.py \
  tests/test_model_server.py \
  tests/test_kmmlu_pro_runner.py \
  tests/test_manifest.py \
  -q
```

## Worktree Notes

- `.external/LiveCodeBench` is untracked and required for local LiveCodeBench runs.
- Many `results/eval_traces/*.jsonl` files are untracked generated outputs.
- No files have been staged or committed.
