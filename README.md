# llm-bench

MLX vs GGUF inference benchmark for local LLMs on Apple Silicon (built for M5 Max 128GB).

Measures **prompt processing speed (PP tok/s)**, **generation speed (TG tok/s)**, **peak memory**, and **output divergence** between two formats of the same model — isolating the runtime, not the model.

First target pair: `gemma-4-26B-A4B-it` MLX 8bit ↔ GGUF Q8_0 (Mixture-of-Experts, 4B active / 26B total).

## Quickstart

```bash
git clone <repo> ~/llm-bench && cd ~/llm-bench

# System tools (one-time)
brew install llama.cpp
brew install --cask quarto      # optional, only for the static report

# Python env
uv sync

# Download GGUF (~27GB). MLX side is loaded from existing HF cache.
bash scripts/download_gguf.sh

# Smoke test (single scenario, ~1 min) — verifies wiring before the full matrix
uv run python scripts/run_bench.py \
  --gguf-model ~/models/gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  --smoke

# Full matrix (~15–25 min on M5 Max)
uv run python scripts/run_bench.py \
  --gguf-model ~/models/gguf/gemma-4-26B-A4B-it-Q8_0.gguf

# Visualize
uv run streamlit run dashboard/app.py

# Output divergence (quality)
uv install --extra quality      # pulls sentence-transformers
uv run python scripts/compare_quality.py \
  --gguf-model ~/models/gguf/gemma-4-26B-A4B-it-Q8_0.gguf

# Static report (requires Quarto)
quarto render report/
open report/_site/index.html
```

## Important: stop other GPU/Metal workloads first

Inference benchmarks are extremely sensitive to Metal contention. Before running:

```bash
# Confirm nothing else is holding the GPU
lsof -i :8080 -i :8081 -i :8082    # mlx servers in ~/llm-stack
ps aux | grep -iE "mlx|llama" | grep -v grep
```

If you run other MLX servers (e.g. `~/llm-stack`), pause them during the run. Otherwise expect 2–5× slower numbers and possible OOM at the 31B class.

## What gets measured

Per (model, format, scenario):

| Metric | Source | Notes |
|---|---|---|
| `pp_tps` (prompt processing tok/s) | mlx-lm verbose / `llama-bench` JSON | Synthetic prefill |
| `tg_tps` (generation tok/s) | mlx-lm verbose / `llama-bench` JSON | Greedy, temp=0 |
| `peak_mem_gb` | `/usr/bin/time -l` max RSS, max'd with `mx.metal.get_peak_memory()` for MLX | Process-level |
| `wall_s` | `/usr/bin/time -l` real | End-to-end including model load |
| `cos_sim` | `paraphrase-multilingual-mpnet-base-v2` embedding | Quality script only |

Scenarios = prefill ∈ {256, 1024, 4096, 8192} × gen ∈ {128, 512}. 3 measured runs + 1 warmup per scenario.

## Repository layout

```
src/llm_bench/
  runners/
    base.py        # BenchResult, time-wrapped subprocess helper
    mlx_runner.py  # invokes mlx_lm.generate
    gguf_runner.py # invokes llama-bench
  prompts.py       # 20 standard quality prompts (KO/EN)
  scenarios.py     # default + smoke matrices
  aggregate.py     # raw JSON → summary CSV
scripts/
  download_gguf.sh
  run_bench.py     # CLI orchestrator
  compare_quality.py
results/
  raw/             # per-run JSON (gitignored)
  summary.csv
  quality_*.json   # gitignored
dashboard/
  app.py           # Streamlit (Overview / Scaling / Quality / Raw)
report/
  _quarto.yml
  index.qmd        # static HTML report
```

## Multi-dimensional evals (added v0.2)

Runs lm-eval-harness against an OpenAI-compatible server (`mlx_lm.server` for
MLX, `llama-server` for GGUF) booted ad-hoc per model variant.

| Dimension | Tasks (chat-compatible) | Loglikelihood-only (gguf only) |
|---|---|---|
| Reasoning | `mmlu_generative`, `gsm8k_cot_zeroshot` | `hellaswag` |
| Korean | `kmmlu_direct`, `hrm8k` | `haerae`, `kobest` |
| Code | `humaneval_instruct`, `mbpp_instruct` | — |
| Long context | `longbench` (21 sub-tasks, EN+ZH) | — |
| Safety | `truthfulqa-multi_gen_en` | `toxigen` |
| Tool use | (BFCL — external repo, optional) | — |

`mlx_lm.server` does not return token logprobs in `/v1/completions`, so
loglikelihood-based MCQ tasks (`hellaswag`, `kobest`, `haerae`, `toxigen`) only
run on the GGUF path. Generative variants are used for the rest so both
runtimes get apples-to-apples coverage.

Setup:

```bash
uv sync --extra evals
```

Smoke (verify wiring, ~10 min, limit=2 per task):

```bash
uv run python scripts/run_evals.py --variant 26B-MoE-mlx-8bit --suite smoke --limit 2
```

Full overnight matrix (all 6 model variants × full suite) — wrapper script
handles launchd bootout + run + bootstrap automatically (always restores agents
on EXIT, even if eval fails):

```bash
# Foreground (watch progress):
bash scripts/run_evals_overnight.sh

# Detached overnight (recommended):
nohup bash scripts/run_evals_overnight.sh > /tmp/llm-evals-overnight.log 2>&1 &
disown
tail -f /tmp/llm-evals-overnight.log
```

Env overrides: `SUITE=smoke`, `LIMIT=10`, `VARIANTS="26B-MoE-mlx-8bit 26B-MoE-gguf-q8"`.

Each variant boots its own server on port 9090; tasks run sequentially per
variant. Expect ~2–3 hours per variant for the full suite.

Results:
- `results/eval_scores/<run_id>/<task>/.../results_*.json` — raw lm-eval output
- `results/eval_scores/summary_*.json` — flat list of {variant, task, results}
- `results/server_logs/<run_id>.log` — model server stderr for debugging

## License

MIT
