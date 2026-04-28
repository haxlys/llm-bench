# Measurement Methodology

## Goals

1. Isolate the **runtime** (MLX vs GGUF/llama.cpp) by holding the model and quantization level constant.
2. Produce numbers a third party can sanity-check against public sources (LLMCheck, llama.cpp discussions).
3. Keep total measurement time under 30 minutes so iteration on tuning (threads, ngl, etc.) is feasible.

## Hardware target

- Apple M5 Max, 128GB unified memory
- macOS 26.4
- AC power, lid open, no Thermal throttling expected within the measurement window

## Software

- Python 3.13 (managed by `uv`)
- `mlx-lm >= 0.20`
- `llama.cpp` from Homebrew (latest stable bottle)
- BSD `time` (`/usr/bin/time -l`) — ships with macOS

## Measurement protocol

For each (runner, scenario):

1. **Warmup**: one full inference pass, results discarded. This populates Metal kernel caches and pages in the model from disk.
2. **N=3 measured runs**, sequential. Each run is a fresh subprocess to avoid in-process caching artifacts.
3. Record per-run: `pp_tps`, `tg_tps`, `peak_mem_gb`, `wall_s`. Stored as JSON in `results/raw/`.
4. Aggregate to `results/summary.csv` with mean and std-dev.

### Why subprocess per run?

MLX and llama.cpp both retain caches across inferences within a single process. Reloading the model each run is honest about cold-load performance and isolates the measurement of *that scenario* from the previous one.

Trade-off: model load (~5–10s for 26B Q8) is included in `wall_s` but **excluded** from `pp_tps` and `tg_tps`, which the underlying tools report based on their own internal timing.

### Why `/usr/bin/time -l` for memory?

It's the only OS-level number that captures peak memory across the whole process lifetime, including Metal allocations that don't show up in `psutil`. It's BSD-only (macOS), which is fine for this project.

For MLX, we cross-reference with `mlx.metal.get_peak_memory()` and take the max.

## Scenario matrix

| Dimension | Values | Rationale |
|---|---|---|
| Prefill length | 256, 1024, 4096, 8192 | Covers chat (256), short doc (1024), medium doc (4096), long doc (8192). Log-spaced. |
| Generation length | 128, 512 | Short reply vs paragraph. |
| Quantization | MLX 8bit, GGUF Q8_0 | Closest pairing in bits/weight. |
| Determinism | temp=0.0 | Removes generation variance. |

8 scenarios × 2 runners × 4 invocations (1 warmup + 3 measured) = **64 invocations** per matrix run.

## Sanity checks

After a measurement run, the dashboard shows:

1. **Std-dev / mean** per scenario. Anything > 5% on `tg_tps` suggests background contention.
2. **Cross-reference**: pull the M5 Max Gemma 4 numbers from LLMCheck. We expect to land within ±15%.
3. **Format sanity**: at the same prefill, MLX and GGUF should have *similar* `pp_tps` curves (both are bound by the same compute), but `tg_tps` may diverge based on memory-bandwidth handling.

## Quality measurement

20 prompts, 8bit MLX vs Q8_0 GGUF, identical prompt + temp=0:

- **word_jaccard** — quick surface metric, sensitive to formatting differences
- **cos_sim** — `paraphrase-multilingual-mpnet-base-v2` embedding cosine. Robust to formatting; sensitive to semantic divergence.
- **Side-by-side raw text** — the most informative signal; the dashboard renders both responses in expanders.

Quality is **not** scored absolutely. We are asking: *does the runtime change the meaning of the output?* Answer is normally "no" (≥0.95 cos_sim) since both load the same weights. Significant divergence would indicate a quantization or numerical-stability bug.

## Multi-dimensional evals (v0.2+)

Beyond speed/memory, accuracy/quality is measured across five dimensions via
lm-eval-harness against an OpenAI-compatible server (`mlx_lm.server` for MLX,
`llama-server` for GGUF) booted ad-hoc per variant on port 9090.

### Tasks per dimension

| Dim | Chat-compatible tasks (works on both fmts) | Loglikelihood-only (GGUF only) |
|---|---|---|
| reasoning | `mmlu_generative`, `gsm8k_cot_zeroshot` | `hellaswag` |
| korean | `kmmlu_direct`, `hrm8k` | `haerae`, `kobest` |
| code | `humaneval_instruct`, `mbpp_instruct` | — |
| long | `longbench` (21 sub-tasks) | — |
| safety | `truthfulqa-multi_gen_en` | `toxigen` |

### Why two task families?

`mlx_lm.server` does not return per-token logprobs in `/v1/completions`
responses. Tasks scored via loglikelihood (multi-choice "which continuation
has higher log P?") therefore cannot run on MLX. We use generative variants
(`mmlu_generative`, `kmmlu_direct`, `truthfulqa-multi_gen_en`) so both
runtimes are evaluated on identical task definitions where possible.

`run_evals.py` calls `supports_fmt(task, fmt)` and prints
`SKIP (loglikelihood-only, fmt=mlx unsupported)` for the four GGUF-only tasks
when the variant is MLX. This is intentional — GGUF gets MCQ coverage, MLX
gets generative-variant coverage of the same underlying ability.

### Eval protocol

For each variant (model_id × fmt × quant):

1. Spawn fresh inference server on port 9090, wait for `/v1/models` 200 OK.
2. Run each task in the suite via `lm_eval` subprocess. Tasks tagged "chat"
   use `local-chat-completions` model class with `--apply_chat_template`;
   loglikelihood tasks use `local-completions` with HF tokenizer.
3. For code-eval tasks, set `HF_ALLOW_CODE_EVAL=1` and pass
   `--confirm_run_unsafe_code` (sandbox-on-trust opt-in for HumanEval/MBPP).
4. Tear down server before next variant.

Results land in `results/eval_scores/<run_id>/<task>/.../results_*.json`.
`scripts/aggregate_evals.py` walks them into two CSVs:

- `eval_summary_full.csv` — every metric × subtask × variant
- `eval_summary_primary.csv` — one row per (variant, task) with the canonical
  headline metric (e.g. `exact_match,strict-match` for gsm8k,
  `pass@1,create_test` for HumanEval, subtask average for hrm8k)

### Registry-driven variants

The list of variants is declared in `models/registry.yaml`, not in code.
Adding a new model = edit YAML, run `sync_models.py`, run `run_bench.py
--all-pending` and `run_evals.py --all-variants --skip-existing`. The
`--skip-existing` flag (default ON) skips combos already measured at the
current `BENCH_VERSION`.

Currently shipped: 6 variants in the gemma-4 family (26B-A4B MoE × {MLX-8bit,
MLX-4bit, Q8_0, Q4_K_M} + 31B Dense × {MLX-8bit, Q8_0}). The full eval
matrix is 6 variants × 12 tasks (minus 4 GGUF-only skips on MLX variants)
≈ 60 invocations. Wall time ≈ 12–18 hours on M5 Max with all production
launchd agents stopped.

### Bench versioning

`src/llm_bench/__init__.py:BENCH_VERSION` is stamped into every BenchResult.
Bumping it triggers full re-measurement on the next `--skip-existing` run.
History: `0.3` is the first version with manifest-based idempotency.

## Out of scope (v0.2)

Done in earlier phases (no longer "out"):
- ✅ 31B Dense model (variant `31B-Dense-*`)
- ✅ 4-bit class (Q4_K_M vs MLX-4bit, variant `*-4bit` / `*-q4`)

Still out of scope:
- Batched / concurrent inference
- KV-cache reuse across turns
- Tool-use / structured output (BFCL skeleton in `evals/bfcl.py`, runner TBD)
- Other model families (Qwen, Llama)
- Power consumption (`powermetrics`)
- Other quantizations (Q5_K_M, Q6_K, IQ4_XS, MLX 6-bit)
- Context lengths beyond 8K (LongBench covers up to 31K avg, but no 64K+)
