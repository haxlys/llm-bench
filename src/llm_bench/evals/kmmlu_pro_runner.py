"""KMMLU-Pro runner using the Hugging Face dataset and OpenAI-compatible chat."""

from __future__ import annotations

import json
import os
import re
import importlib.util
from hashlib import sha1
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "LGAI-EXAONE/KMMLU-Pro"
DEFAULT_CONFIG = "kmmlu_pro"
DEFAULT_TASK_TIMEOUT_S = 2 * 60 * 60


def run_kmmlu_pro(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TASK_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Evaluate KMMLU-Pro with deterministic multiple-choice extraction."""
    del timeout_s  # Direct SDK calls currently rely on the OpenAI client's timeout defaults.
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "kmmlu_pro.log"

    try:
        rows = _load_rows(limit)
    except Exception as e:
        return {
            "task": "kmmlu_pro",
            "error": f"dataset load failed: {e}",
            "log": str(log_path),
        }
    if not rows:
        return {"task": "kmmlu_pro", "error": "no KMMLU-Pro rows loaded", "log": str(log_path)}

    try:
        client = _openai_client(base_url=base_url, api_key=api_key)
    except Exception as e:
        return {"task": "kmmlu_pro", "error": f"openai client unavailable: {e}", "log": str(log_path)}

    samples_path = output_dir / "samples_kmmlu_pro.jsonl"
    scored: list[dict[str, Any]] = []
    errors: list[str] = []
    with samples_path.open("w", encoding="utf-8") as f, log_path.open("w", encoding="utf-8") as log:
        for idx, row in enumerate(rows):
            if idx % 100 == 0:
                log.write(f"progress {idx}/{len(rows)}\n")
                log.flush()
            prompt = _build_prompt(row)
            try:
                response = client.chat.completions.create(
                    model=model_label,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=int(os.environ.get("KMMLU_PRO_MAX_TOKENS", "512")),
                )
                answer_text = response.choices[0].message.content or ""
            except Exception as e:
                errors.append(f"row {idx}: {e}")
                log.write(errors[-1] + "\n")
                log.flush()
                answer_text = ""

            extracted = _extract_choice(answer_text)
            target = _target_choice(row)
            is_correct = extracted == target
            record = {
                "idx": idx,
                "question": row.get("question", ""),
                "license_name": row.get("license_name", ""),
                "subject": row.get("subject", ""),
                "target": target,
                "response": answer_text,
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "score": _weight(row),
            }
            scored.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        log.write(f"progress {len(rows)}/{len(rows)}\n")

    weighted_acc = _weighted_accuracy(scored)
    metrics = {"acc,none": weighted_acc, "questions,none": float(len(scored))}
    for license_name, value in _license_scores(scored).items():
        metrics[f"{license_name}_acc,none"] = value

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    synthetic_path = output_dir / f"results_{ts}_kmmlu_pro.json"
    synthetic_path.write_text(
        json.dumps({"results": {"kmmlu_pro": metrics}}, ensure_ascii=False)
    )

    result = {
        "task": "kmmlu_pro",
        "results_file": str(synthetic_path),
        "samples_file": str(samples_path),
        "results": {"kmmlu_pro": metrics},
    }
    if errors:
        result["warnings"] = errors[:5]
    return result


def kmmlu_pro_available() -> bool:
    return _module_exists("datasets") and _module_exists("openai")


def _load_rows(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split="test")
    rows = list(dataset)
    return rows[:limit] if limit is not None else rows


def _openai_client(base_url: str, api_key: str | None):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "local-no-auth"),
        base_url=base_url.rstrip("/"),
    )


def _build_prompt(row: dict[str, Any]) -> str:
    options = _options(row)
    option_lines = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    return (
        "다음은 한국어 전문자격시험 객관식 문제입니다.\n"
        "정답 선택지는 A, B, C, D 또는 E 중 하나입니다.\n"
        "마지막 줄에 반드시 `정답: <선택지>` 형식으로 답하세요.\n\n"
        f"문제:\n{row.get('question', '')}\n\n"
        f"선택지:\n{option_lines}"
    )


def _options(row: dict[str, Any]) -> list[str]:
    raw = row.get("options")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    out = []
    for key in ("A", "B", "C", "D", "E"):
        if key in row:
            out.append(str(row[key]))
    return out


def _target_choice(row: dict[str, Any]) -> str:
    raw = row.get("solution", row.get("answer", row.get("target", "")))
    if isinstance(raw, str):
        stripped = raw.strip().upper()
        if stripped in {"A", "B", "C", "D", "E"}:
            return stripped
        if stripped.isdigit():
            return _choice_from_index(int(stripped))
    if isinstance(raw, (int, float)):
        return _choice_from_index(int(raw))
    return ""


def _choice_from_index(index: int) -> str:
    if 1 <= index <= 5:
        return chr(64 + index)
    if 0 <= index <= 4:
        return chr(65 + index)
    return ""


def _extract_choice(text: str) -> str:
    patterns = [
        r"(?:정답|답|answer)\s*[:：]?\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).upper()
    return ""


def _weight(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _weighted_accuracy(records: list[dict[str, Any]]) -> float:
    total = sum(float(r.get("score", 1.0)) for r in records)
    if total <= 0:
        return 0.0
    correct = sum(float(r.get("score", 1.0)) for r in records if r.get("is_correct"))
    return correct / total


def _license_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        raw = str(record.get("license_name") or "").strip()
        if not raw:
            continue
        grouped.setdefault(_metric_slug(raw), []).append(record)
    return {license_name: _weighted_accuracy(items) for license_name, items in grouped.items()}


def _metric_slug(value: str) -> str:
    ascii_slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if ascii_slug:
        return ascii_slug
    # Keep Korean labels deterministic without putting non-ASCII in metric names.
    return "license_" + sha1(value.encode("utf-8")).hexdigest()[:10]


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False
