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

## Out of scope (v0.1)

- Batched / concurrent inference
- KV-cache reuse across turns
- Tool-use / structured output
- 31B Dense model (planned for v0.2)
- 4bit class (Q4_K_M vs MLX-4bit)
- Other models (Qwen, Llama)
- Power consumption (powermetrics)
