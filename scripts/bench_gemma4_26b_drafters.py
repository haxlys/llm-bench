"""Benchmark Gemma 4 26B A4B MTP drafters on local MLX."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm.generate import generate
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import load


ROOT = Path(__file__).resolve().parent.parent
HF_HUB = Path("/Users/haxlys/models/huggingface/hub")

TARGETS = [
    (
        "26B-A4B-4bit",
        HF_HUB / (
            "models--lmstudio-community--gemma-4-26B-A4B-it-MLX-4bit/"
            "snapshots/3af5252ed1e675e6bba9be8cc3087bc00920799c"
        ),
    ),
    (
        "26B-A4B-8bit",
        HF_HUB / (
            "models--lmstudio-community--gemma-4-26B-A4B-it-MLX-8bit/"
            "snapshots/669237cd8dad363224c976432475b81dd5db5a89"
        ),
    ),
]

DRAFTERS = [
    (
        "google/gemma-4-26B-A4B-it-assistant",
        HF_HUB / (
            "models--google--gemma-4-26B-A4B-it-assistant/"
            "snapshots/b7696b57a62c7a24a177254d1135fd6c0e650792"
        ),
    ),
    (
        "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
        HF_HUB / (
            "models--mlx-community--gemma-4-26B-A4B-it-assistant-bf16/"
            "snapshots/cda74908f1dbe7d3dbd3030e66576a7d4094144f"
        ),
    ),
]

PROMPT = (
    "Write a concise technical explanation of speculative decoding for local "
    "LLM inference. Include why it can improve tokens per second and one "
    "caveat. Continue until the answer is complete."
)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _reset_metal_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def _settle() -> None:
    mx.eval(mx.array([0]))
    mx.clear_cache()
    gc.collect()


def _summarize(label: str, trials: list[dict[str, Any]], draft_model: Any | None = None) -> dict[str, Any]:
    out = {
        "label": label,
        "trials": len(trials),
        "generation_tps": [t["generation_tps"] for t in trials],
        "generation_tps_mean": _mean([t["generation_tps"] for t in trials]),
        "prompt_tps_mean": _mean([t.get("prompt_tps", 0.0) for t in trials]),
        "tokens": [t["tokens"] for t in trials],
        "wall_seconds": [t["wall_seconds"] for t in trials],
        "wall_seconds_mean": _mean([t["wall_seconds"] for t in trials]),
        "peak_memory_gb_max": max([t.get("peak_memory_gb", 0.0) for t in trials] or [0.0]),
        "raw_trials": trials,
    }
    accept_lens = getattr(draft_model, "accept_lens", None) if draft_model is not None else None
    if accept_lens:
        out["accepted_tokens_per_round_mean"] = _mean([float(x) for x in accept_lens])
        out["accepted_rounds"] = len(accept_lens)
    return out


def _run_trial(model: Any, processor: Any, max_tokens: int, draft_model: Any | None = None) -> dict[str, Any]:
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
    result = generate(model, processor, PROMPT, **kwargs)
    _settle()
    return {
        "tokens": int(getattr(result, "generation_tokens", 0)),
        "generation_tps": float(getattr(result, "generation_tps", 0.0)),
        "prompt_tps": float(getattr(result, "prompt_tps", 0.0)),
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_gb": float(getattr(result, "peak_memory", mx.get_peak_memory() / 1e9)),
    }


def _run_trials(
    model: Any,
    processor: Any,
    label: str,
    max_tokens: int,
    trials: int,
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

    print(f"[mtp] warmup {label} max_tokens={max_tokens}", flush=True)
    generate(model, processor, PROMPT, **warmup_kwargs)
    _settle()

    if draft_model is not None and hasattr(draft_model, "accept_lens"):
        draft_model.accept_lens = []

    raw: list[dict[str, Any]] = []
    for idx in range(trials):
        print(f"[mtp] trial {idx + 1}/{trials} {label} max_tokens={max_tokens}", flush=True)
        raw.append(_run_trial(model, processor, max_tokens, draft_model))
    return _summarize(label, raw, draft_model)


def run_target(label: str, target_path: Path, max_tokens_values: list[int], trials: int) -> dict[str, Any]:
    print(f"[mtp] loading target {label}: {target_path}", flush=True)
    model, processor = load(str(target_path))
    rows: list[dict[str, Any]] = []

    for max_tokens in max_tokens_values:
        baseline = _run_trials(model, processor, f"{label}-baseline", max_tokens, trials)
        row: dict[str, Any] = {
            "max_tokens": max_tokens,
            "baseline": baseline,
            "drafters": [],
        }

        for drafter_label, drafter_path in DRAFTERS:
            print(f"[mtp] loading drafter {drafter_label}: {drafter_path}", flush=True)
            drafter = load_drafter(str(drafter_path), kind="mtp")
            drafted = _run_trials(model, processor, drafter_label, max_tokens, trials, drafter)
            base_tps = baseline["generation_tps_mean"]
            drafted["speedup_vs_baseline"] = (
                drafted["generation_tps_mean"] / base_tps if base_tps else 0.0
            )
            row["drafters"].append(drafted)
            del drafter
            _settle()

        rows.append(row)

    del model, processor
    _settle()
    return {
        "target_label": label,
        "target_model": str(target_path),
        "draft_block_size": 6,
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--target", choices=["4bit", "8bit", "all"], default="all")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/Users/haxlys/models/huggingface")
    selected = [
        item for item in TARGETS
        if args.target == "all" or item[0].endswith(args.target)
    ]

    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "created_at": started.isoformat(),
        "host": os.uname().nodename,
        "runtime": "mlx_vlm",
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "temperature": 0.0,
        "targets": [],
    }

    for label, target_path in selected:
        report["targets"].append(run_target(label, target_path, args.max_tokens, args.trials))

    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["wall_seconds"] = (finished - started).total_seconds()

    out = args.out
    if out is None:
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"gemma4_26b_drafter_benchmark_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"RESULT_PATH={out}", flush=True)


if __name__ == "__main__":
    main()
