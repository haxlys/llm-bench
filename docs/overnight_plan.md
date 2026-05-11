# Overnight Eval Catch-up Plan

Updated: 2026-05-11

Goal: complete the minimum common primary matrix first, then run optional
frontier lanes separately so the site never blends missing coverage into model
quality claims.

Regenerate the concrete queue from the current coverage index:

```bash
uv run python scripts/plan_eval_catchup.py
```

This writes `results/eval_catchup_plan.json` and
`results/eval_catchup_plan.md` with task-filtered commands.

## Primary matrix

Run with resilient IFEval and strict coverage:

```bash
LLM_BENCH_RESILIENT_IFEVAL=1 \
LLM_BENCH_STRICT_COVERAGE=1 \
SUITE=full \
bash scripts/run_evals_overnight.sh
```

Recommended task buckets:

```bash
# 1. Source-grounding + Korean professional coverage
TASKS="sourceqa kmmlu_pro" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh

# 2. Primary code coverage
TASKS="humaneval mbpp livecodebench" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh

# 3. Remaining reasoning / Korean / instruction coverage
TASKS="gsm8k_cot_zeroshot hrm8k leaderboard_ifeval" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh
```

Primary tasks to fill where supported:

| Lane | Tasks |
|---|---|
| Reasoning / instruction | `gsm8k_cot_zeroshot`, `hrm8k`, `leaderboard_ifeval` |
| Korean | `kmmlu_pro` |
| Code | `humaneval`, `mbpp`, `livecodebench` |
| Source grounding | `sourceqa` |
| Speed | all `results/summary.csv` scenarios via `scripts/run_bench.py --all-pending` |
| MTPLX speedup | `scripts/compare_mtplx.py` after paired MTPLX speed runs |

MTPLX MTP/AR variants are speed-only. They are visible as `mtplx_speedup` /
`speed_only` rows in coverage and are skipped by `run_evals.py --all-variants`.

## Family batches

Use `VARIANTS=...` to keep each run reviewable:

```bash
# Open-weight reasoning comparison
VARIANTS="gpt-oss-20b-gguf-q4 gpt-oss-120b-gguf-q4" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh

# Code-specialized comparison
VARIANTS="qwen-3-coder-30b-a3b-instruct-gguf-q4 qwen-3-coder-next-gguf-q4" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh

# MoE / throughput comparison
VARIANTS="qwen-3-next-80b-a3b-instruct-gguf-q4 qwen-3.6-35b-a3b-gguf-q4 nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4" \
LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 \
bash scripts/run_evals_overnight.sh
```

## Version pins

- LiveCodeBench: default `release_v6`; set `LIVE_CODE_BENCH_RELEASE=release_v6`
  explicitly for release runs. Keep upstream fast/lite default unless a release
  gate needs `LIVE_CODE_BENCH_NOT_FAST=1`.
- LiveBench: default `LIVEBENCH_RELEASE=2024-11-25` because upstream notes newer
  releases may not be fully public across categories.
- BFCL: BFCL V4, official leaderboard checkpoint/package
  `bfcl-eval==2025.12.17`.
- BigCodeBench: split `instruct`, subset `hard`.
- ProgramBench: record the upstream ProgramBench package/release next to the
  agent submission run before importing `*.eval.json`.

## Optional lanes

Run optional lanes after primary coverage is green:

```bash
# BFCL explicitly opt-in
TASKS="bfcl" LLM_BENCH_INCLUDE_BFCL=1 \
bash scripts/run_evals_overnight.sh

# BigCodeBench-Hard and LiveBench subset stay separate from the primary matrix
TASKS="bigcodebench_hard livebench_subset" \
bash scripts/run_evals_overnight.sh

# ProgramBench import after agent submissions exist
uv run python scripts/import_programbench.py \
  --variant qwen-3-coder-next-gguf-q4 \
  --source-dir /path/to/programbench/evaluated-run \
  --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
```

Finish every batch with:

```bash
uv run python scripts/aggregate_evals.py
uv run python scripts/export_site_public_data.py
```
