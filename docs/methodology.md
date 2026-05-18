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

Beyond speed/memory, accuracy/quality is measured across multiple dimensions
via lm-eval-harness and lightweight external runners against an
OpenAI-compatible server (`mlx_lm.server` for MLX, `llama-server` for GGUF)
booted ad-hoc per local variant on port 9090. Hosted endpoint variants reuse
their configured `/v1` endpoint directly and do not spawn a subprocess.

### Tasks per dimension

| Dim | lm-eval tasks | External runners |
|---|---|---|
| reasoning | `leaderboard_mmlu_pro`, `leaderboard_gpqa_diamond`, `mmlu_generative`, `gsm8k_cot_zeroshot` | — |
| korean | `kmmlu_direct`, `hrm8k` | `kmmlu_pro` |
| instruction | `leaderboard_ifeval` | — |
| safety | `truthfulqa-multi_gen_en`, `toxigen` | — |
| diagnostic source grounding | — | `sourceqa` (pinned-repo evidence QA) |
| code / practical generation | — | primary: `humaneval`, `mbpp` (EvalPlus), `bigcodebench_hard`; optional: `livecodebench`, `livebench_subset` |
| tool use | — | optional: `bfcl` (opt-in via `--include-bfcl`) |
| long context | `longbench` (21 sub-tasks, run with `--suite long`) | — |
| agentic code | — | primary: `terminal_bench`; optional: `programbench` eval + result import |

### Why two task families?

`mlx_lm.server` does not return per-token logprobs in `/v1/completions`
responses. Loglikelihood-only tasks (`leaderboard_mmlu_pro`, `leaderboard_gpqa_diamond`,
`toxigen`, etc.) still run only on logprobs-capable backends (GGUF). We use
generative variants (`mmlu_generative`, `kmmlu_direct`, `truthfulqa-multi_gen_en`)
so both runtimes are evaluated on comparable task definitions where possible.

`run_evals.py` gates tasks through declared variant capabilities. For example,
tasks that require logprobs are skipped for backends that only declare chat and
completion support.

### Eval protocol

For each variant (model_id × fmt × quant):

1. Spawn a fresh local inference server on port 9090, or reuse a configured endpoint.
2. Run each task in the suite via `lm_eval` subprocess. Tasks tagged "chat"
   use `local-chat-completions` model class with `--apply_chat_template`;
   loglikelihood tasks use `local-completions` with HF tokenizer.
3. For code-eval external runners (`EvalPlus`, `LiveCodeBench`,
   `BigCodeBench-Hard`, `Terminal-Bench`), required toolchains are wrapped in
   their own runner environments. They can still execute user code or Docker
   workloads, so use trusted checkpoints for safety and keep runner-level
   sandboxing in mind.
4. Tear down server before next variant.

Results land in `results/eval_scores/<run_id>/<task>/.../results_*.json`.
`scripts/aggregate_evals.py` walks them into two CSVs:

- `eval_summary_full.csv` — every metric × subtask × variant
- `eval_summary_primary.csv` — one row per (variant, task) with the canonical
  headline metric (e.g. `exact_match,strict-match` for gsm8k,
  `pass@1,create_test` for HumanEval, subtask average for hrm8k)
- `index.json` — registry × speed/eval coverage, including whether each task is
  `measured`, `directional`, `diagnostic`, `missing`, `optional`,
  `speed_only`, or `unsupported`

External runners also emit aggregate-compatible synthetic `results_*.json`
files. For example, EvalPlus keeps its native `*_eval_results.json` beside the
samples, then writes a compact `results_*.json` with `pass_at_1,base` and
`pass_at_1,plus` so the same CSV/index path can consume it.

`sourceqa` is a source-grounding diagnostic, not a headline ranking dimension.
Each task declares a pinned repo, commit, question, `required_any` signals,
`forbidden` phrases, and exact `evidence_paths`. The runner clones the pinned
commit, injects only the curated evidence files into the prompt, and scores the
answer deterministically via required-signal recall, evidence-path recall, and
forbidden-phrase violations. Optional judge metadata can be requested, but it is
recorded separately and does not change the `acc,none` metric. Because the
current task set is small and saturated, SourceQA is kept for smoke/regression
checks and excluded from headline ranking and primary coverage debt.

`programbench` is the agentic whole-program reconstruction dimension.
ProgramBench's Docker evaluation targets Linux x86_64 infrastructure and the
agent scaffold materially affects results. llm-bench therefore separates agent
submission generation from scoring: once an agent has produced
`<instance_id>/submission.tar.gz`, `scripts/run_programbench.py` invokes
upstream `programbench eval` and imports the resulting `*.eval.json` files into
the same synthetic `results_*.json` shape as other external runners. If scoring
already happened elsewhere, `scripts/import_programbench.py` can import those
eval JSON files directly. The primary metric is `resolved_rate,none` (fully
solved instances). `almost_resolved_rate,none` and `avg_test_pass_rate,none`
are supporting diagnostics. When a ProgramBench `data/tasks` directory is
provided, ignored branches/tests from `tests.json` are excluded to mirror
`programbench info`.

`terminal_bench` is the primary maintained agentic terminal-task dimension. llm-bench
invokes upstream `tb run` against the same OpenAI-compatible endpoint and then
imports Terminal-Bench `results.json` into synthetic results with primary metric
`resolved_rate,none`. It is Docker-backed; the default wrapper path runs one
task unless `TERMINAL_BENCH_N_TASKS`, `TERMINAL_BENCH_TASK_IDS`, or
`TERMINAL_BENCH_FULL=1` is set.

Every eval task execution also appends one row to
`results/eval_traces/<run_id>.jsonl` with variant, task, runner, status, wall
time, result artifact path, optional sample path, log path, and error text.

### Coverage lanes and score status

Primary lanes are the minimum common matrix used for model-family comparisons:
reasoning (`gsm8k_cot_zeroshot`, instruction), Korean (`hrm8k`, `kmmlu_pro`),
code (`humaneval`, `mbpp`, `bigcodebench_hard`), agentic code
(`terminal_bench`), and speed where applicable.
SourceQA is a diagnostic lane for source-grounding smoke/regression checks, not
primary coverage debt. MTPLX native MTP/AR variants are a separate
`mtplx_speedup` lane, not accuracy candidates. Optional lanes are intentionally separated:
`livecodebench`, `bfcl`, `livebench_subset`, and `programbench`.

The site reads `results/index.json` before showing score tables. That lets it
distinguish:

- `measured`: committed result with a direct deterministic score.
- `directional`: committed result whose scorer can undercount because of
  generation formatting or extraction.
- `diagnostic`: committed diagnostic result that remains visible but is excluded
  from headline ranking and primary coverage debt.
- `missing`: primary supported task with no committed result yet.
- `optional`: optional-lane task with no committed result yet.
- `speed_only`: MTPLX speedup variant; accuracy coverage is intentionally not
  required for that row.
- `unsupported`: task is not valid for the variant's declared capabilities.

`--strict-coverage` fails only on missing primary supported tasks. Optional
lanes, diagnostic rows, and MTPLX speed-only rows remain visible but do not
block a primary matrix run. Use the generated `results/eval_catchup_plan.md`
commands to fill catch-up buckets in the planned order.

### Registry-driven variants

The list of variants is declared in `models/registry.yaml`, not in code.
Adding a new model = edit YAML, run `sync_models.py`, run `run_bench.py
--all-pending` and `run_evals.py --all-variants --skip-existing`. The
`--skip-existing` flag (default ON) skips combos already measured at the
current `BENCH_VERSION`.

`run_evals.py --all-variants` skips `backend: mtplx` variants because those
entries are speed-only speculative decoding comparators. The paired flat MLX
variants stay in the quality/eval matrix.

Registry variants now distinguish the legacy `fmt` label from the more general
runtime `backend`, `artifact_type`, `capabilities`, `api_model`, and
`api_key_env`. Existing MLX/GGUF
entries infer these fields automatically (`mlx` → `hf_repo` without logprobs,
`gguf` → `gguf_file` with logprobs). Hosted or remote models can be described
with `fmt: api`, `backend: openai-compatible`, `artifact_type: endpoint`, and
explicit capabilities such as `chat`, `completions`, `logprobs`, or
`tool_use_eval`. Speed runners cover MLX, GGUF, and OpenAI-compatible
endpoints; endpoint speed uses wall-clock effective token rates because hosted
APIs generally do not expose separate prefill/generation timings. Unsupported
backends fail explicitly instead of being treated as GGUF. Eval runners can use
`openai-compatible` endpoint variants directly; those variants reuse the
configured endpoint instead of booting a local subprocess.

Current shipped matrix now includes Gemma, Qwen/Qwen3.6, gpt-oss, Nemotron,
and MTPLX variants alongside hosted endpoints. The full eval matrix size depends
on enabled external runners and format support; GGUF adds additional
loglikelihood-only coverage while MLX uses chat-compatible counterparts.
Wall time is driven mostly by long-context and frontier external runners, so run
those only when long-context and frontier capability coverage are the explicit
goal.

### Bench versioning

`src/llm_bench/__init__.py:BENCH_VERSION` is stamped into every BenchResult.
Bumping it triggers full re-measurement on the next `--skip-existing` run.
History: `0.3` is the first version with manifest-based idempotency.

External benchmark versions are pinned separately because their upstream data
changes over time:

- LiveCodeBench defaults to `release_v6`; override with
  `LIVE_CODE_BENCH_RELEASE=release_vX`. The runner uses the upstream default
  fast/lite code-generation setting unless `LIVE_CODE_BENCH_NOT_FAST=1` is set.
- LiveBench defaults to `2024-11-25`, the most recent fully public release noted
  by upstream for all categories. Override with `LIVEBENCH_RELEASE=YYYY-MM-DD`.
- BFCL is treated as BFCL V4 and should use the official reproducibility
  checkpoint/package (`bfcl-eval==2025.12.17`, leaderboard updated 2026-04-12).
- BigCodeBench is pinned in code to split `instruct`, subset `hard`.
- ProgramBench is imported from explicit `*.eval.json` artifacts; record the
  upstream ProgramBench package/release beside the agent run when generating
  those submissions.
- Terminal-Bench defaults to `terminal-bench-core==0.1.1` through
  `terminal-bench>=0.2.18`; override with `TERMINAL_BENCH_DATASET`.

## Out of scope (v0.2)

Done in earlier phases (no longer "out"):
- ✅ 31B Dense model (variant `31B-Dense-*`)
- ✅ 4-bit class (Q4_K_M vs MLX-4bit, variant `*-4bit` / `*-q4`)

Still out of scope:
- Batched / concurrent inference
- KV-cache reuse across turns
- Power consumption (`powermetrics`)
- Other quantizations (Q5_K_M, Q6_K, IQ4_XS, MLX 6-bit)
- Context lengths beyond 31K+ (LongBench covers up to 31K avg, not 64K+)
