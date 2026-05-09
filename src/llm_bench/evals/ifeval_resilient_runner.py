"""Resilient leaderboard IFEval runner for local OpenAI-compatible servers."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "wis-k/instruction-following-eval"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_TOKENS = 512


def run_leaderboard_ifeval_resilient(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Run leaderboard_ifeval while scoring per-sample API failures as empty.

    lm-eval aborts the whole task when llama-server returns a transient HTTP 500
    for one generation. For local GGUF runs this runner keeps the official
    IFEval scoring utilities but records failed samples as empty responses so
    the benchmark can finish and the failure remains visible in samples/logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "leaderboard_ifeval_resilient.log"
    samples_path = output_dir / "samples_leaderboard_ifeval_resilient.jsonl"
    max_tokens = max_tokens or int(os.environ.get("IFEVAL_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    try:
        rows = _load_rows(limit)
        client = _openai_client(base_url=base_url, api_key=api_key)
        scorer = _ifeval_scorer()
    except Exception as e:
        return {"task": "leaderboard_ifeval", "error": str(e), "log": str(log_path)}

    prompt_strict: list[bool] = []
    prompt_loose: list[bool] = []
    inst_strict: list[bool] = []
    inst_loose: list[bool] = []
    errors: list[str] = []

    with log_path.open("w", encoding="utf-8") as log, samples_path.open("w", encoding="utf-8") as samples:
        for idx, row in enumerate(rows):
            if idx % 25 == 0:
                log.write(f"progress {idx}/{len(rows)}\n")
                log.flush()

            response_text, error = _generate(
                client=client,
                model_label=model_label,
                prompt=str(row["prompt"]),
                max_tokens=max_tokens,
            )
            if error:
                errors.append(f"row {idx} key={row.get('key')}: {error}")
                log.write(errors[-1] + "\n")
                log.flush()

            metrics = scorer.process_results(row, [response_text])
            prompt_strict.append(bool(metrics["prompt_level_strict_acc"]))
            prompt_loose.append(bool(metrics["prompt_level_loose_acc"]))
            inst_strict.extend(bool(x) for x in metrics["inst_level_strict_acc"])
            inst_loose.extend(bool(x) for x in metrics["inst_level_loose_acc"])

            samples.write(
                json.dumps(
                    {
                        "idx": idx,
                        "key": row.get("key"),
                        "prompt": row.get("prompt"),
                        "response": response_text,
                        "error": error,
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            samples.flush()

        log.write(f"progress {len(rows)}/{len(rows)}\n")

    result_metrics = {
        "prompt_level_strict_acc,none": _mean(prompt_strict),
        "inst_level_strict_acc,none": _mean(inst_strict),
        "prompt_level_loose_acc,none": _mean(prompt_loose),
        "inst_level_loose_acc,none": _mean(inst_loose),
        "prompt_level_strict_acc_stderr,none": _stderr(prompt_strict),
        "inst_level_strict_acc_stderr,none": _stderr(inst_strict),
        "prompt_level_loose_acc_stderr,none": _stderr(prompt_loose),
        "inst_level_loose_acc_stderr,none": _stderr(inst_loose),
        "samples,none": float(len(rows)),
        "api_errors,none": float(len(errors)),
    }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    results_path = output_dir / f"results_{ts}_leaderboard_ifeval.json"
    results_path.write_text(
        json.dumps({"results": {"leaderboard_ifeval": result_metrics}}, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "task": "leaderboard_ifeval",
        "results_file": str(results_path),
        "samples_file": str(samples_path),
        "log": str(log_path),
        "results": {"leaderboard_ifeval": result_metrics},
        "warnings": errors[:5],
    }


def _load_rows(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DEFAULT_DATASET, split=DEFAULT_SPLIT)
    rows = list(dataset)
    return rows[:limit] if limit is not None else rows


def _openai_client(base_url: str, api_key: str | None):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "local-no-auth"),
        base_url=base_url.rstrip("/"),
        max_retries=0,
        timeout=600,
    )


def _ifeval_scorer():
    from lm_eval.tasks.leaderboard.ifeval import utils

    return utils


def _generate(client, model_label: str, prompt: str, max_tokens: int) -> tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=model_label,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or "", ""
    except Exception as chat_error:
        try:
            response = client.completions.create(
                model=model_label,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].text or "", f"chat failed; completions fallback used: {chat_error}"
        except Exception as completion_error:
            return "", f"chat failed: {chat_error}; completions failed: {completion_error}"


def _mean(values: list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stderr(values: list[bool]) -> float:
    if not values:
        return 0.0
    p = _mean(values)
    return float(math.sqrt(p * (1.0 - p) / len(values)))
