"""Walk lm-eval-harness output and build a tidy DataFrame of all metrics.

Schema of the resulting frame (`load_eval_results`):

    variant     str   "26B-MoE-mlx-8bit"
    model_id    str   "gemma-4-26B-A4B-it"
    fmt         str   "mlx" | "gguf"
    quant       str   "MLX-8bit" | "Q8_0" | ...
    tier        str   "8bit" | "4bit" (mapped from quant)
    dim         str   "reasoning" | "korean" | "code" | "long" | "safety"
    task        str   top-level task in suites.SUITES
    subtask     str   sub-result key from results JSON ("" for top-level only)
    metric      str   metric column name from lm-eval ("acc,none", "pass@1,create_test", ...)
    value       float
    stderr      float | NaN
    run_id      str   directory name in results/eval_scores/
    ts          str   parsed timestamp from run_id

Helper views:
    primary_metric_view(df) — one row per (variant, task) with the canonical
                              headline number, easy to plot as heatmap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# Reuse VARIANTS metadata + tier map from existing modules.
TIER_MAP = {"MLX-8bit": "8bit", "Q8_0": "8bit", "MLX-4bit": "4bit", "Q4_K_M": "4bit"}

# Variant key → (model_id, fmt, quant). Mirrors VARIANTS in scripts/run_evals.py.
VARIANT_META = {
    "26B-MoE-mlx-8bit":   ("gemma-4-26B-A4B-it", "mlx",  "MLX-8bit"),
    "26B-MoE-mlx-4bit":   ("gemma-4-26B-A4B-it", "mlx",  "MLX-4bit"),
    "26B-MoE-gguf-q8":    ("gemma-4-26B-A4B-it", "gguf", "Q8_0"),
    "26B-MoE-gguf-q4":    ("gemma-4-26B-A4B-it", "gguf", "Q4_K_M"),
    "31B-Dense-mlx-8bit": ("gemma-4-31B-it",     "mlx",  "MLX-8bit"),
    "31B-Dense-gguf-q8":  ("gemma-4-31B-it",     "gguf", "Q8_0"),
}

# Run-id directory name follows pattern: <ts>_<variant>_<suite>
RUN_DIR_RE = re.compile(r"^(?P<ts>\d{8}T\d{6}Z)_(?P<variant>[\w-]+?)_(?P<suite>smoke|full)$")

# Top-level task → dim mapping (mirrors suites.SUITES).
TASK_DIM = {
    "mmlu_generative": "reasoning",
    "gsm8k_cot_zeroshot": "reasoning",
    "hellaswag": "reasoning",
    "kmmlu_direct": "korean",
    "hrm8k": "korean",
    "haerae": "korean",
    "kobest": "korean",
    "humaneval_instruct": "code",
    "mbpp_instruct": "code",
    "longbench": "long",
    "truthfulqa-multi_gen_en": "safety",
    "toxigen": "safety",
}

# Heuristic: which metric is the headline number for each task.
# Order matters — first match wins.
PRIMARY_METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,get_response",
    "exact_match,none",
    "pass@1,create_test",
    "pass@1,none",
    "acc,none",
    "acc_norm,none",
    "score,none",
    "rouge_score,none",
    "f1,none",
    "qa_f1_score,none",
    "bleu_max,none",
    "bleurt_max,none",
]

NUMERIC_SUFFIXES = (",none", ",strict-match", ",flexible-extract", ",create_test")


def _parse_run_dir(name: str) -> dict | None:
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    return m.groupdict()


def _flatten_results_json(path: Path) -> list[dict]:
    """Each results_*.json may aggregate multiple subtasks. Return one row per metric."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    results = data.get("results", {})
    rows: list[dict] = []
    for subtask, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if k == "alias" or not isinstance(v, (int, float)):
                continue
            if k.endswith("_stderr"):  # paired with non-stderr below
                continue
            stderr_key = k.split(",")[0] + "_stderr," + k.split(",", 1)[1] if "," in k else None
            stderr = metrics.get(stderr_key) if stderr_key else None
            rows.append({
                "subtask": subtask,
                "metric": k,
                "value": float(v),
                "stderr": float(stderr) if isinstance(stderr, (int, float)) else float("nan"),
            })
    return rows


def load_eval_results(eval_dir: Path) -> pd.DataFrame:
    """Walk eval_dir and load all results into a tidy DataFrame."""
    rows: list[dict] = []
    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _parse_run_dir(run_dir.name)
        if not meta:
            continue
        variant = meta["variant"]
        ts = meta["ts"]
        v_meta = VARIANT_META.get(variant)
        if not v_meta:
            continue
        model_id, fmt, quant = v_meta
        tier = TIER_MAP.get(quant, "")
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            dim = TASK_DIM.get(task, "other")
            for results_file in task_dir.rglob("results_*.json"):
                for r in _flatten_results_json(results_file):
                    rows.append({
                        "variant": variant, "model_id": model_id, "fmt": fmt,
                        "quant": quant, "tier": tier, "dim": dim, "task": task,
                        "run_id": run_dir.name, "ts": ts, **r,
                    })
    return pd.DataFrame(rows)


def primary_metric_view(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one canonical metric per (variant, task). Returns one row each.

    Strategy:
      1. Prefer a row where subtask == task (the lm-eval aggregate row).
      2. If no such row exists (e.g. tasks like hrm8k that only emit subtask
         rows), average across subtasks per metric and treat as the headline.
    """
    if df.empty:
        return df

    out_rows: list[pd.Series] = []
    for (variant, task), group in df.groupby(["variant", "task"], sort=False):
        top = group[group["subtask"] == task]
        if top.empty:
            # average across subtasks for each metric
            agg = (group.groupby("metric", as_index=False)
                   .agg(value=("value", "mean"), stderr=("stderr", "mean")))
            agg["variant"] = variant
            agg["task"] = task
            agg["model_id"] = group["model_id"].iloc[0]
            agg["fmt"] = group["fmt"].iloc[0]
            agg["quant"] = group["quant"].iloc[0]
            agg["tier"] = group["tier"].iloc[0]
            agg["dim"] = group["dim"].iloc[0]
            agg["subtask"] = "<aggregated>"
            top = agg

        chosen = None
        for m in PRIMARY_METRICS:
            hit = top[top["metric"] == m]
            if not hit.empty:
                chosen = hit.iloc[0]
                break
        if chosen is None:
            chosen = top.iloc[0]
        out_rows.append(chosen)

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).reset_index(drop=True)


def write_summary(eval_dir: Path, out_csv: Path) -> Path:
    df = load_eval_results(eval_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
