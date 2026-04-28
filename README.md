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
uv sync --extra quality          # pulls sentence-transformers
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

## Model registry (`models/registry.yaml`)

The single source of truth for what gets benchmarked. Adding a new model:

```yaml
models:
  - id: qwen-3.6-27b
    family: qwen
    architecture: dense
    params_total_b: 27
    variants:
      - key: qwen-27b-mlx-8bit
        fmt: mlx
        path: mlx-community/Qwen3.6-27B-8bit
        quant: MLX-8bit
        tier: 8bit
        approx_size_gb: 27
      - key: qwen-27b-gguf-q8
        fmt: gguf
        path: "{gguf_dir}/qwen-3.6-27b-Q8_0.gguf"
        quant: Q8_0
        tier: 8bit
        approx_size_gb: 28
        download:
          repo: bartowski/qwen-3.6-27b-GGUF
          pattern: "*Q8_0*.gguf"
```

Then:

```bash
uv run python scripts/sync_models.py --model qwen-3.6-27b
uv run python scripts/run_bench.py --variant qwen-27b-mlx-8bit --variant qwen-27b-gguf-q8
uv run python scripts/run_evals.py --variant qwen-27b-mlx-8bit --variant qwen-27b-gguf-q8 --suite full
```

Currently shipped (six variants, gemma-4 family):

| Key | Model | Format | Quant | Tier |
|---|---|---|---|---|
| `26B-MoE-mlx-8bit`   | gemma-4-26B-A4B-it (MoE) | MLX  | 8-bit  | 8bit |
| `26B-MoE-gguf-q8`    | gemma-4-26B-A4B-it (MoE) | GGUF | Q8_0   | 8bit |
| `26B-MoE-mlx-4bit`   | gemma-4-26B-A4B-it (MoE) | MLX  | 4-bit  | 4bit |
| `26B-MoE-gguf-q4`    | gemma-4-26B-A4B-it (MoE) | GGUF | Q4_K_M | 4bit |
| `31B-Dense-mlx-8bit` | gemma-4-31B-it (Dense)   | MLX  | 8-bit  | 8bit |
| `31B-Dense-gguf-q8`  | gemma-4-31B-it (Dense)   | GGUF | Q8_0   | 8bit |

`tier` pairs MLX-Nbit ↔ Q*_K_M for fair runtime comparisons. The dashboard
**Catalog** page shows registry × measurement status at a glance.

## Idempotency

Every measurement records the current `bench_version` (currently `0.3`).
`run_bench.py --skip-existing` (default ON) skips combos that already have N
runs at that version; `--all-pending` runs only what's missing across the
registry. Bumping `BENCH_VERSION` in `src/llm_bench/__init__.py` triggers a
full re-measurement when methodology changes.

## Repository layout

```
models/
  registry.yaml             # single source of truth — add a model here
src/llm_bench/
  __init__.py               # BENCH_VERSION constant
  registry.py               # YAML loader + Variant/Model dataclasses
  manifest.py               # idempotency: which (variant, scenario) is measured
  index.py                  # build results/index.json (registry × status)
  runners/                  # speed/memory benchmark
    base.py                 # BenchResult + /usr/bin/time -l wrapper
    mlx_runner.py           # mlx_lm.generate subprocess
    gguf_runner.py          # llama-bench subprocess
  evals/                    # multi-dim accuracy (lm-eval-harness)
    server.py               # ModelServer (mlx_lm.server | llama-server)
    lmeval.py               # lm_eval CLI wrapper
    suites.py               # SMOKE/FULL task lists, supports_fmt()
    aggregate.py            # eval JSON → tidy DataFrame
    bfcl.py                 # external-repo placeholder
  prompts.py                # 20 quality-comparison prompts (KO/EN)
  scenarios.py              # speed scenario matrices
  aggregate.py              # speed raw JSON → summary CSV
scripts/
  sync_models.py            # registry-driven hf download
  run_bench.py              # speed CLI: --variant / --all-pending
  run_evals.py              # eval CLI: --variant / --all-variants
  run_evals_overnight.sh    # launchd stop → run → aggregate → restore
  compare_quality.py        # cos-sim divergence (20 prompts)
  aggregate_evals.py        # eval JSON → CSVs + index.json
  build_index.py            # build only the index
results/
  raw/                      # per-run speed JSON (gitignored)
  summary.csv               # speed aggregated (committed)
  quality_*.json            # gitignored
  eval_scores/              # lm-eval outputs (gitignored)
  eval_summary_*.csv        # eval aggregated (committed)
  index.json                # registry × measurement status (committed)
  server_logs/              # gitignored
  overnight_logs/           # gitignored
dashboard/
  app.py                    # Streamlit (11 pages, Catalog first)
report/
  _quarto.yml
  index.qmd                 # static HTML report (Quarto)
docs/
  methodology.md            # measurement protocol
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
manages optional launchd bootout + run + bootstrap automatically (always
restores agents on EXIT, even if eval fails):

```bash
# Foreground (watch progress):
bash scripts/run_evals_overnight.sh

# Detached overnight (recommended):
nohup bash scripts/run_evals_overnight.sh > /tmp/llm-evals-overnight.log 2>&1 &
disown
tail -f /tmp/llm-evals-overnight.log
```

Env overrides:
- `SUITE=smoke|full` (default `full`)
- `LIMIT=N` (per-task sample cap)
- `VARIANTS="26B-MoE-mlx-8bit 26B-MoE-gguf-q8"` (subset, default = all)
- `LAUNCH_AGENTS="com.you.foo com.you.bar"` — launchd agent labels to stop
  before the run and restart at the end. Default empty = no launchd
  management; stop GPU-using processes manually instead.

Each variant boots its own server on port 9090; tasks run sequentially per
variant. Expect ~2–3 hours per variant for the full suite.

Results:
- `results/eval_scores/<run_id>/<task>/.../results_*.json` — raw lm-eval output
- `results/eval_scores/summary_*.json` — flat list of {variant, task, results}
- `results/eval_summary_full.csv` — every metric × subtask × variant (244+ rows)
- `results/eval_summary_primary.csv` — one row per (variant, task), canonical metric
- `results/server_logs/<run_id>.log` — model server stderr for debugging

After the eval run, `scripts/aggregate_evals.py` rebuilds the CSVs and the
Streamlit dashboard auto-loads them. The overnight wrapper calls this for you.

## Dashboard (11 pages)

```bash
uv run streamlit run dashboard/app.py
```

| Group | Page | What it shows |
|---|---|---|
| Status | **Catalog** | Registry × measurement progress bars (entry point) |
| Speed | **Speed Overview** | TG/PP bar charts, peak memory, Pareto scatter |
| Speed | **Speed Scaling** | Context-length sweep with format colors |
| Speed | **Output Quality (cos sim)** | 20-prompt MLX-vs-GGUF response similarity |
| Speed | **Speed Raw** | Per-run JSON table + CSV download |
| Eval | **Evals Heatmap** | Variant × task primary-score grid |
| Eval | **Evals · MLX vs GGUF** | Score delta within same model+tier |
| Eval | **Evals · Quantization** | 8bit vs 4bit accuracy hit per (model, fmt) |
| Eval | **Evals · Dimension** | Per-dim bar charts with stderr |
| Eval | **Evals · LongBench Detail** | 21 sub-task breakdown |
| Eval | **Evals Raw** | Full metrics filterable table + CSV |

## Methodology

See [docs/methodology.md](docs/methodology.md) for measurement protocol,
sanity checks, scenario matrix rationale, and the chat-vs-loglikelihood split.

## License

MIT
