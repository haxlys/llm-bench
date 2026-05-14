# Eval Catch-up Plan

Generated from `results/index.json`.

## Summary

- Primary missing rows: 28
- Optional pending rows: 66
- Speed-incomplete variants: 7

## Primary evals

### kmmlu_pro

Fill Korean professional coverage first.

- Tasks: kmmlu_pro
- Variants: 1

```bash
VARIANTS="deepseek-v4-flash-gguf-iq2xxs" TASKS="kmmlu_pro" SUITE=full LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 bash scripts/run_evals_overnight.sh
```

### primary_code

Fill standard code coverage before optional agentic/code lanes.

- Tasks: humaneval, mbpp, livecodebench
- Variants: 9

```bash
VARIANTS="31B-Dense-gguf-q8 deepseek-v4-flash-gguf-iq2xxs gpt-oss-120b-gguf-q4 gpt-oss-20b-gguf-q4 nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4 qwen-3-coder-30b-a3b-instruct-gguf-q4 qwen-3-next-80b-a3b-instruct-gguf-q4 qwen-3.5-9b-gguf-q8 qwen-3.6-35b-a3b-gguf-q4" TASKS="humaneval mbpp livecodebench" SUITE=full LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 bash scripts/run_evals_overnight.sh
```

### reasoning_instruction_catchup

Catch remaining reasoning, Korean, and instruction-following primary gaps.

- Tasks: gsm8k_cot_zeroshot, hrm8k, leaderboard_ifeval
- Variants: 5

```bash
VARIANTS="26B-MoE-gguf-q4 26B-MoE-mlx-4bit 31B-Dense-gguf-q8 31B-Dense-mlx-8bit deepseek-v4-flash-gguf-iq2xxs" TASKS="gsm8k_cot_zeroshot hrm8k leaderboard_ifeval" SUITE=full LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_STRICT_COVERAGE=1 bash scripts/run_evals_overnight.sh
```

## Speed matrix

Complete speed matrix and regenerate MTPLX speedups.

- deepseek-v4-flash-gguf-iq2xxs: 0/8 scenarios
- qwen-3.6-27b-mtplx-optimized-mlx-mixed4: 0/8 scenarios
- qwen-3.6-27b-mtplx-optimized-mtplx-ar: 0/8 scenarios
- qwen-3.6-27b-mtplx-optimized-mtplx-mtp: 0/8 scenarios
- qwen-3.6-27b-mtplx-speed-mlx-4bit: 0/8 scenarios
- qwen-3.6-27b-mtplx-speed-mtplx-ar: 4/8 scenarios
- qwen-3.6-27b-mtplx-speed-mtplx-mtp: 4/8 scenarios

```bash
uv run python scripts/run_bench.py --skip-existing --variant deepseek-v4-flash-gguf-iq2xxs --variant qwen-3.6-27b-mtplx-optimized-mlx-mixed4 --variant qwen-3.6-27b-mtplx-optimized-mtplx-ar --variant qwen-3.6-27b-mtplx-optimized-mtplx-mtp --variant qwen-3.6-27b-mtplx-speed-mlx-4bit --variant qwen-3.6-27b-mtplx-speed-mtplx-ar --variant qwen-3.6-27b-mtplx-speed-mtplx-mtp
uv run python scripts/compare_mtplx.py && uv run python scripts/aggregate_evals.py && uv run python scripts/export_site_public_data.py
```

## Optional lanes

### optional_bigcodebench_bfcl_livebench

Run optional frontier lanes separately after primary coverage.

- Tasks: bigcodebench_hard, bfcl, livebench_subset
- Variants: 19

```bash
VARIANTS="26B-MoE-gguf-q4 26B-MoE-gguf-q8 26B-MoE-mlx-4bit 26B-MoE-mlx-8bit 31B-Dense-gguf-q8 31B-Dense-mlx-8bit deepseek-v4-flash-gguf-iq2xxs gemma-4-E4B-gguf-q8 gpt-oss-120b-gguf-q4 gpt-oss-20b-gguf-q4 nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4 qwen-3-coder-30b-a3b-instruct-gguf-q4 qwen-3-coder-next-gguf-q4 qwen-3-next-80b-a3b-instruct-gguf-q4 qwen-3.5-4b-gguf-q8 qwen-3.5-9b-gguf-q8 qwen-3.6-27b-mtplx-optimized-mlx-mixed4 qwen-3.6-27b-mtplx-speed-mlx-4bit qwen-3.6-35b-a3b-gguf-q4" TASKS="bigcodebench_hard bfcl livebench_subset" SUITE=full LLM_BENCH_RESILIENT_IFEVAL=1 LLM_BENCH_INCLUDE_BFCL=1 bash scripts/run_evals_overnight.sh
```

### optional_programbench

Import ProgramBench agent submission results after external runs complete.

- Tasks: programbench
- Variants: 19

```bash
# ProgramBench requires completed agent submissions/eval JSONs first.
uv run python scripts/import_programbench.py --variant 26B-MoE-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant 26B-MoE-gguf-q8 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant 26B-MoE-mlx-4bit --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant 26B-MoE-mlx-8bit --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant 31B-Dense-gguf-q8 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant 31B-Dense-mlx-8bit --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant deepseek-v4-flash-gguf-iq2xxs --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant gemma-4-E4B-gguf-q8 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant gpt-oss-120b-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant gpt-oss-20b-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant nemotron-3-nano-omni-30b-a3b-reasoning-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3-coder-30b-a3b-instruct-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3-coder-next-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3-next-80b-a3b-instruct-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3.5-4b-gguf-q8 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3.5-9b-gguf-q8 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3.6-27b-mtplx-optimized-mlx-mixed4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3.6-27b-mtplx-speed-mlx-4bit --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
uv run python scripts/import_programbench.py --variant qwen-3.6-35b-a3b-gguf-q4 --source-dir /path/to/programbench/evaluated-run --tasks-dir /path/to/ProgramBench/src/programbench/data/tasks
```
