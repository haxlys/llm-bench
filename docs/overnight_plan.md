# Overnight Eval Catch-up Plan

Updated: 2026-05-11

Goal: complete the minimum common primary matrix first, then run optional
frontier lanes separately so the site never blends missing coverage into model
quality claims.

## Primary matrix

Run with resilient IFEval and strict coverage:

```bash
LLM_BENCH_RESILIENT_IFEVAL=1 \
LLM_BENCH_STRICT_COVERAGE=1 \
SUITE=full \
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
uv run python scripts/run_evals.py --variant qwen-3-coder-next-gguf-q4 \
  --suite full --include-bfcl --skip-existing

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
