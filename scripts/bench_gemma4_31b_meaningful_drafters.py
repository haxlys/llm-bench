"""Meaningful Gemma 4 31B drafter benchmark across prompt types.

The quick one-prompt smoke benches are useful for wiring checks, but DFlash and
MTP acceptance are prompt-sensitive. This script runs multiple prompt families
and records per-prompt as well as aggregate speedups.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import stream_generate as stream_generate_baseline
from mlx_lm.sample_utils import make_sampler
from mlx_vlm.generate import generate
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import load as load_vlm


ROOT = Path(__file__).resolve().parent.parent
HF_HUB = Path("/Users/haxlys/models/huggingface/hub")
DFLASH_VENDOR = Path(
    "/Users/haxlys/Documents/Codex/2026-05-07/"
    "https-huggingface-co-z-lab-gemma/vendor/dflash"
)

TARGET_31B_4BIT = HF_HUB / (
    "models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05"
)
GOOGLE_ASSISTANT_31B = HF_HUB / (
    "models--google--gemma-4-31B-it-assistant/"
    "snapshots/cffbbd2cea41ea56a0fa5b0487e0d445121fd204"
)
MLX_ASSISTANT_31B = HF_HUB / (
    "models--mlx-community--gemma-4-31B-it-assistant-bf16/"
    "snapshots/28e92270316e89288579ec59c17939541d9ca433"
)
DFLASH_TARGET_REPO = "mlx-community/gemma-4-31b-it-4bit"
DFLASH_DRAFT_REPO = "z-lab/gemma-4-31B-it-DFlash"


@dataclass(frozen=True)
class PromptCase:
    key: str
    label: str
    text: str


PROMPTS = [
    PromptCase(
        "technical",
        "Technical explanation",
        (
            "Write a concise technical explanation of speculative decoding for "
            "local LLM inference. Include why it can improve tokens per second, "
            "how acceptance affects speed, and one caveat."
        ),
    ),
    PromptCase(
        "code",
        "Code generation",
        (
            "Write a Python implementation of merge sort with type hints. Then "
            "explain its time complexity, space complexity, and one practical "
            "optimization for nearly sorted input."
        ),
    ),
    PromptCase(
        "math",
        "Math reasoning",
        (
            "Solve this carefully: A data pipeline processes 18,000 records per "
            "minute. A speculative decoder reduces generation latency by 35%, "
            "but adds 8% verification overhead. If the original end-to-end job "
            "takes 42 minutes and generation is 60% of runtime, estimate the new "
            "runtime and show the calculation."
        ),
    ),
    PromptCase(
        "korean",
        "Korean explanation",
        (
            "한국어로 설명해 주세요. 온디바이스 LLM에서 speculative decoding과 "
            "MTP drafter가 어떻게 토큰 생성 속도를 높이는지, 그리고 acceptance가 "
            "낮을 때 왜 오히려 느려질 수 있는지 예시와 함께 정리하세요."
        ),
    ),
]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "median": _median(values),
        "stdev": _stdev(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _reset_metal_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def _settle() -> None:
    mx.eval(mx.array([0]))
    mx.clear_cache()
    gc.collect()


def _summarize_trials(label: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    tps = [float(t["generation_tps"]) for t in trials]
    wall = [float(t["wall_seconds"]) for t in trials]
    prompt_tps = [float(t.get("prompt_tps", 0.0)) for t in trials]
    return {
        "label": label,
        "trials": len(trials),
        "generation_tps": tps,
        "generation_tps_stats": _stats(tps),
        "generation_tps_mean": _mean(tps),
        "generation_tps_median": _median(tps),
        "prompt_tps_mean": _mean(prompt_tps),
        "tokens": [int(t["tokens"]) for t in trials],
        "wall_seconds": wall,
        "wall_seconds_stats": _stats(wall),
        "wall_seconds_mean": _mean(wall),
        "peak_memory_gb_max": max([float(t.get("peak_memory_gb", 0.0)) for t in trials] or [0.0]),
        "raw_trials": trials,
    }


def _aggregate_mode(prompt_rows: list[dict[str, Any]], mode_key: str) -> dict[str, Any]:
    speedups = [float(row[mode_key]["speedup_vs_baseline"]) for row in prompt_rows]
    tps = [float(row[mode_key]["generation_tps_mean"]) for row in prompt_rows]
    accept = [
        float(row[mode_key].get("accepted_tokens_per_round_mean", row[mode_key].get("avg_accept_length_mean", 0.0)))
        for row in prompt_rows
    ]
    return {
        "speedup_stats": _stats(speedups),
        "generation_tps_stats": _stats(tps),
        "acceptance_stats": _stats(accept),
        "prompt_count": len(prompt_rows),
    }


def _mtp_trial(model: Any, processor: Any, prompt: str, max_tokens: int, draft_model: Any | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "verbose": False,
    }
    if draft_model is not None:
        kwargs.update(
            {
                "draft_model": draft_model,
                "draft_kind": "mtp",
                "draft_block_size": 6,
            }
        )

    _reset_metal_peak()
    started = time.perf_counter()
    result = generate(model, processor, prompt, **kwargs)
    _settle()
    return {
        "tokens": int(getattr(result, "generation_tokens", 0)),
        "generation_tps": float(getattr(result, "generation_tps", 0.0)),
        "prompt_tps": float(getattr(result, "prompt_tps", 0.0)),
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_gb": float(getattr(result, "peak_memory", mx.get_peak_memory() / 1e9)),
    }


def _run_mtp_prompt(
    model: Any,
    processor: Any,
    prompt_case: PromptCase,
    max_tokens: int,
    trials: int,
    label: str,
    draft_model: Any | None = None,
) -> dict[str, Any]:
    warmup_kwargs: dict[str, Any] = {
        "max_tokens": min(16, max_tokens),
        "temperature": 0.0,
        "verbose": False,
    }
    if draft_model is not None:
        warmup_kwargs.update(
            {
                "draft_model": draft_model,
                "draft_kind": "mtp",
                "draft_block_size": 6,
            }
        )

    print(f"[mtp] warmup {label} prompt={prompt_case.key}", flush=True)
    generate(model, processor, prompt_case.text, **warmup_kwargs)
    _settle()

    if draft_model is not None and hasattr(draft_model, "accept_lens"):
        draft_model.accept_lens = []

    raw = []
    for idx in range(trials):
        print(f"[mtp] trial {idx + 1}/{trials} {label} prompt={prompt_case.key}", flush=True)
        raw.append(_mtp_trial(model, processor, prompt_case.text, max_tokens, draft_model))

    out = _summarize_trials(label, raw)
    accept_lens = getattr(draft_model, "accept_lens", None) if draft_model is not None else None
    if accept_lens:
        out["accepted_tokens_per_round_mean"] = _mean([float(x) for x in accept_lens])
        out["accepted_rounds"] = len(accept_lens)
    return out


def run_mtp_suite(prompt_cases: list[PromptCase], max_tokens: int, trials: int) -> dict[str, Any]:
    print(f"[mtp] loading target {TARGET_31B_4BIT}", flush=True)
    model, processor = load_vlm(str(TARGET_31B_4BIT))

    prompt_rows: list[dict[str, Any]] = []
    for prompt_case in prompt_cases:
        baseline = _run_mtp_prompt(model, processor, prompt_case, max_tokens, trials, "baseline")
        row: dict[str, Any] = {
            "prompt_key": prompt_case.key,
            "prompt_label": prompt_case.label,
            "baseline": baseline,
        }

        for mode_key, label, path in [
            ("google_assistant", "google/gemma-4-31B-it-assistant", GOOGLE_ASSISTANT_31B),
            ("mlx_assistant", "mlx-community/gemma-4-31B-it-assistant-bf16", MLX_ASSISTANT_31B),
        ]:
            print(f"[mtp] loading drafter {label}", flush=True)
            drafter = load_drafter(str(path), kind="mtp")
            result = _run_mtp_prompt(model, processor, prompt_case, max_tokens, trials, label, drafter)
            base_tps = baseline["generation_tps_mean"]
            result["speedup_vs_baseline"] = result["generation_tps_mean"] / base_tps if base_tps else 0.0
            row[mode_key] = result
            del drafter
            _settle()

        prompt_rows.append(row)

    del model, processor
    _settle()
    return {
        "runtime": "mlx_vlm",
        "target_model": str(TARGET_31B_4BIT),
        "draft_block_size": 6,
        "prompts": prompt_rows,
        "aggregate": {
            "google_assistant": _aggregate_mode(prompt_rows, "google_assistant"),
            "mlx_assistant": _aggregate_mode(prompt_rows, "mlx_assistant"),
        },
    }


def _load_dflash_symbols() -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(DFLASH_VENDOR))
    from dflash.model_mlx import load, load_draft, stream_generate

    return load, load_draft, stream_generate


def _dflash_prompt(tokenizer: Any, prompt_case: PromptCase) -> str:
    messages = [{"role": "user", "content": prompt_case.text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _dflash_baseline_trial(model: Any, tokenizer: Any, prompt: str, max_tokens: int, sampler: Any) -> dict[str, Any]:
    token_count = 0
    final_tps = 0.0
    _reset_metal_peak()
    started = time.perf_counter()
    for response in stream_generate_baseline(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        token_count += 1
        final_tps = float(response.generation_tps)
    _settle()
    return {
        "tokens": token_count,
        "generation_tps": final_tps,
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def _dflash_trial(
    model: Any,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    sampler: Any,
    stream_generate: Any,
) -> dict[str, Any]:
    token_count = 0
    accepted_lengths: list[int] = []
    final_tps = 0.0
    peak_memory = 0.0
    _reset_metal_peak()
    started = time.perf_counter()
    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt,
        block_size=16,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        token_count += len(response.tokens)
        if response.accepted:
            accepted_lengths.append(int(response.accepted))
        final_tps = float(response.generation_tps)
        peak_memory = float(response.peak_memory)
    _settle()
    return {
        "tokens": token_count,
        "generation_tps": final_tps,
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory or mx.get_peak_memory() / 1e9,
        "avg_accept_length": _mean([float(x) for x in accepted_lengths]),
        "accept_lengths": accepted_lengths,
    }


def _run_dflash_baseline_prompt(
    model: Any,
    tokenizer: Any,
    prompt_case: PromptCase,
    max_tokens: int,
    trials: int,
    sampler: Any,
) -> dict[str, Any]:
    prompt = _dflash_prompt(tokenizer, prompt_case)
    print(f"[dflash] baseline warmup prompt={prompt_case.key}", flush=True)
    list(stream_generate_baseline(model, tokenizer, tokenizer.encode("Hi"), 4, sampler=sampler))
    _settle()

    raw = []
    for idx in range(trials):
        print(f"[dflash] baseline trial {idx + 1}/{trials} prompt={prompt_case.key}", flush=True)
        raw.append(_dflash_baseline_trial(model, tokenizer, prompt, max_tokens, sampler))
    return _summarize_trials("baseline", raw)


def _run_dflash_prompt(
    model: Any,
    draft: Any,
    tokenizer: Any,
    prompt_case: PromptCase,
    max_tokens: int,
    trials: int,
    sampler: Any,
    stream_generate: Any,
) -> dict[str, Any]:
    prompt = _dflash_prompt(tokenizer, prompt_case)
    raw = []
    for idx in range(trials):
        print(f"[dflash] trial {idx + 1}/{trials} prompt={prompt_case.key}", flush=True)
        raw.append(_dflash_trial(model, draft, tokenizer, prompt, max_tokens, sampler, stream_generate))

    out = _summarize_trials(DFLASH_DRAFT_REPO, raw)
    out["avg_accept_length_mean"] = _mean([float(t.get("avg_accept_length", 0.0)) for t in raw])
    return out


def run_dflash_suite(prompt_cases: list[PromptCase], max_tokens: int, trials: int) -> dict[str, Any]:
    load, load_draft, stream_generate = _load_dflash_symbols()

    print(f"[dflash] loading target {DFLASH_TARGET_REPO}", flush=True)
    model, tokenizer = load(DFLASH_TARGET_REPO)
    sampler = make_sampler(temp=0.0)

    prompt_rows: list[dict[str, Any]] = []
    for prompt_case in prompt_cases:
        baseline = _run_dflash_baseline_prompt(model, tokenizer, prompt_case, max_tokens, trials, sampler)
        prompt_rows.append(
            {
                "prompt_key": prompt_case.key,
                "prompt_label": prompt_case.label,
                "baseline": baseline,
            }
        )

    print(f"[dflash] loading draft {DFLASH_DRAFT_REPO}", flush=True)
    draft = load_draft(DFLASH_DRAFT_REPO)
    print("[dflash] draft warmup", flush=True)
    list(stream_generate(model, draft, tokenizer, tokenizer.encode("Hi"), block_size=16, max_tokens=4, sampler=sampler))
    _settle()

    for row, prompt_case in zip(prompt_rows, prompt_cases):
        drafted = _run_dflash_prompt(model, draft, tokenizer, prompt_case, max_tokens, trials, sampler, stream_generate)
        base_tps = row["baseline"]["generation_tps_mean"]
        drafted["speedup_vs_baseline"] = drafted["generation_tps_mean"] / base_tps if base_tps else 0.0
        row["dflash"] = drafted

    del model, tokenizer, draft
    _settle()
    return {
        "runtime": "dflash_mlx",
        "target_model": DFLASH_TARGET_REPO,
        "draft_block_size": 16,
        "prompts": prompt_rows,
        "aggregate": {
            "dflash": _aggregate_mode(prompt_rows, "dflash"),
        },
    }


def _write_summary_csv(report: dict[str, Any], out_path: Path) -> Path:
    csv_path = out_path.with_suffix(".summary.csv")
    rows: list[dict[str, Any]] = []
    for suite in report["suites"]:
        for row in suite["prompts"]:
            baseline = row["baseline"]
            for mode_key, mode in row.items():
                if mode_key in {"prompt_key", "prompt_label", "baseline"}:
                    continue
                rows.append(
                    {
                        "runtime": suite["runtime"],
                        "prompt_key": row["prompt_key"],
                        "prompt_label": row["prompt_label"],
                        "mode": mode["label"],
                        "baseline_tps_mean": baseline["generation_tps_mean"],
                        "mode_tps_mean": mode["generation_tps_mean"],
                        "speedup_vs_baseline": mode["speedup_vs_baseline"],
                        "acceptance_mean": mode.get(
                            "accepted_tokens_per_round_mean",
                            mode.get("avg_accept_length_mean", ""),
                        ),
                        "mode_peak_memory_gb_max": mode["peak_memory_gb_max"],
                    }
                )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", choices=[p.key for p in PROMPTS], default=[p.key for p in PROMPTS])
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--skip-dflash", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/Users/haxlys/models/huggingface")
    prompt_cases = [p for p in PROMPTS if p.key in set(args.prompts)]
    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "created_at": started.isoformat(),
        "host": os.uname().nodename,
        "benchmark_kind": "meaningful_gemma4_31b_drafters",
        "max_tokens": args.max_tokens,
        "trials_per_prompt": args.trials,
        "prompt_count": len(prompt_cases),
        "temperature": 0.0,
        "prompt_cases": [p.__dict__ for p in prompt_cases],
        "suites": [],
        "errors": [],
    }

    if not args.skip_mtp:
        try:
            report["suites"].append(run_mtp_suite(prompt_cases, args.max_tokens, args.trials))
        except Exception:
            report["errors"].append({"suite": "mtp", "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)

    if not args.skip_dflash:
        try:
            report["suites"].append(run_dflash_suite(prompt_cases, args.max_tokens, args.trials))
        except Exception:
            report["errors"].append({"suite": "dflash", "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)

    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["wall_seconds"] = (finished - started).total_seconds()

    out = args.out
    if out is None:
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"gemma4_31b_meaningful_drafter_benchmark_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = _write_summary_csv(report, out)
    print(json.dumps(report, indent=2), flush=True)
    print(f"RESULT_PATH={out}", flush=True)
    print(f"SUMMARY_CSV={csv_path}", flush=True)


if __name__ == "__main__":
    main()
