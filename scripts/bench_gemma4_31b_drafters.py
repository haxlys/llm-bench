"""Benchmark Gemma 4 31B speculative drafters on local MLX.

This is intentionally outside the registry path because Gemma 4 MTP and
DFlash currently use different experimental runtimes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
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

TARGET_4BIT_SNAPSHOT = HF_HUB / (
    "models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05"
)
TARGET_8BIT_SNAPSHOT = HF_HUB / (
    "models--lmstudio-community--gemma-4-31B-it-MLX-8bit/"
    "snapshots/244e29d3b19e7b50e3ddddc33fcc882f24a19399"
)
GOOGLE_ASSISTANT_SNAPSHOT = HF_HUB / (
    "models--google--gemma-4-31B-it-assistant/"
    "snapshots/cffbbd2cea41ea56a0fa5b0487e0d445121fd204"
)
MLX_ASSISTANT_SNAPSHOT = HF_HUB / (
    "models--mlx-community--gemma-4-31B-it-assistant-bf16/"
    "snapshots/28e92270316e89288579ec59c17939541d9ca433"
)
DFLASH_DRAFT_REPO = "z-lab/gemma-4-31B-it-DFlash"
DFLASH_TARGET_REPO = "mlx-community/gemma-4-31b-it-4bit"

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


def _summarize_trials(label: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
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


def _mtp_trial(model: Any, processor: Any, max_tokens: int, draft_model: Any | None) -> dict[str, Any]:
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


def _run_mtp_trials(
    model: Any,
    processor: Any,
    label: str,
    max_tokens: int,
    trials: int,
    draft_model: Any | None = None,
) -> dict[str, Any]:
    print(f"[mtp] warmup {label} max_tokens={max_tokens}", flush=True)
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
    generate(model, processor, PROMPT, **warmup_kwargs)
    _settle()

    if draft_model is not None and hasattr(draft_model, "accept_lens"):
        draft_model.accept_lens = []

    raw: list[dict[str, Any]] = []
    for idx in range(trials):
        print(f"[mtp] trial {idx + 1}/{trials} {label} max_tokens={max_tokens}", flush=True)
        raw.append(_mtp_trial(model, processor, max_tokens, draft_model))

    out = _summarize_trials(label, raw)
    accept_lens = getattr(draft_model, "accept_lens", None) if draft_model is not None else None
    if accept_lens:
        out["accepted_tokens_per_round_mean"] = _mean([float(x) for x in accept_lens])
        out["accepted_rounds"] = len(accept_lens)
    return out


def run_mtp_suite(max_tokens_values: list[int], trials: int, target_snapshot: Path) -> dict[str, Any]:
    print(f"[mtp] loading target {target_snapshot}", flush=True)
    model, processor = load_vlm(str(target_snapshot))

    results: list[dict[str, Any]] = []
    for max_tokens in max_tokens_values:
        baseline = _run_mtp_trials(model, processor, "baseline", max_tokens, trials)
        row: dict[str, Any] = {
            "max_tokens": max_tokens,
            "baseline": baseline,
            "drafters": [],
        }

        for label, path in [
            ("google/gemma-4-31B-it-assistant", GOOGLE_ASSISTANT_SNAPSHOT),
            ("mlx-community/gemma-4-31B-it-assistant-bf16", MLX_ASSISTANT_SNAPSHOT),
        ]:
            print(f"[mtp] loading drafter {label}: {path}", flush=True)
            drafter = load_drafter(str(path), kind="mtp")
            drafted = _run_mtp_trials(model, processor, label, max_tokens, trials, drafter)
            base_tps = baseline["generation_tps_mean"]
            drafted["speedup_vs_baseline"] = (
                drafted["generation_tps_mean"] / base_tps if base_tps else 0.0
            )
            row["drafters"].append(drafted)
            del drafter
            _settle()

        results.append(row)

    del model, processor
    _settle()
    return {
        "runtime": "mlx_vlm",
        "target_model": str(target_snapshot),
        "draft_block_size": 6,
        "results": results,
    }


def _load_dflash_symbols() -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(DFLASH_VENDOR))
    from dflash.model_mlx import load, load_draft, stream_generate

    return load, load_draft, stream_generate


def _dflash_prompt(tokenizer: Any) -> str:
    messages = [{"role": "user", "content": PROMPT}]
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


def run_dflash_suite(max_tokens_values: list[int], trials: int) -> dict[str, Any]:
    load, load_draft, stream_generate = _load_dflash_symbols()

    print(f"[dflash] loading target {DFLASH_TARGET_REPO}", flush=True)
    model, tokenizer = load(DFLASH_TARGET_REPO)
    prompt = _dflash_prompt(tokenizer)
    sampler = make_sampler(temp=0.0)

    print("[dflash] baseline warmup", flush=True)
    list(stream_generate_baseline(model, tokenizer, tokenizer.encode("Hi"), 4, sampler=sampler))
    _settle()

    rows: list[dict[str, Any]] = []
    for max_tokens in max_tokens_values:
        baseline_trials = []
        for idx in range(trials):
            print(f"[dflash] baseline trial {idx + 1}/{trials} max_tokens={max_tokens}", flush=True)
            baseline_trials.append(_dflash_baseline_trial(model, tokenizer, prompt, max_tokens, sampler))

        rows.append(
            {
                "max_tokens": max_tokens,
                "baseline": _summarize_trials("baseline", baseline_trials),
            }
        )

    print(f"[dflash] loading draft {DFLASH_DRAFT_REPO}", flush=True)
    draft = load_draft(DFLASH_DRAFT_REPO)
    print("[dflash] draft warmup", flush=True)
    list(stream_generate(model, draft, tokenizer, tokenizer.encode("Hi"), block_size=16, max_tokens=4, sampler=sampler))
    _settle()

    for row in rows:
        max_tokens = int(row["max_tokens"])
        draft_trials = []
        for idx in range(trials):
            print(f"[dflash] trial {idx + 1}/{trials} max_tokens={max_tokens}", flush=True)
            draft_trials.append(
                _dflash_trial(model, draft, tokenizer, prompt, max_tokens, sampler, stream_generate)
            )
        drafted = _summarize_trials(DFLASH_DRAFT_REPO, draft_trials)
        drafted["avg_accept_length_mean"] = _mean(
            [t.get("avg_accept_length", 0.0) for t in draft_trials]
        )
        base_tps = row["baseline"]["generation_tps_mean"]
        drafted["speedup_vs_baseline"] = (
            drafted["generation_tps_mean"] / base_tps if base_tps else 0.0
        )
        row["drafters"] = [drafted]

    del model, tokenizer, draft
    _settle()
    return {
        "runtime": "dflash_mlx",
        "target_model": DFLASH_TARGET_REPO,
        "draft_block_size": 16,
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--skip-dflash", action="store_true")
    parser.add_argument("--mtp-target", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/Users/haxlys/models/huggingface")
    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "created_at": started.isoformat(),
        "host": os.uname().nodename,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "temperature": 0.0,
        "suites": [],
        "errors": [],
    }

    if not args.skip_mtp:
        try:
            target = TARGET_8BIT_SNAPSHOT if args.mtp_target == "8bit" else TARGET_4BIT_SNAPSHOT
            report["suites"].append(run_mtp_suite(args.max_tokens, args.trials, target))
        except Exception:
            report["errors"].append({"suite": "mtp", "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)

    if not args.skip_dflash:
        try:
            report["suites"].append(run_dflash_suite(args.max_tokens, args.trials))
        except Exception:
            report["errors"].append({"suite": "dflash", "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)

    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["wall_seconds"] = (finished - started).total_seconds()

    out = args.out
    if out is None:
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"gemma4_31b_drafter_benchmark_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"RESULT_PATH={out}", flush=True)


if __name__ == "__main__":
    main()
