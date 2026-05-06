"""Export committed benchmark artifacts into static website data."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from llm_bench.registry import Registry, Variant, load_registry


class SiteDataError(ValueError):
    """Raised when benchmark artifacts cannot be exported safely."""


_SCENARIO_RE = re.compile(r"^p(?P<prompt>\d+)_g(?P<generation>\d+)$")
_DIRECTIONAL_TASKS = {
    "gsm8k_cot_zeroshot",
    "hrm8k",
    "kmmlu_direct",
    "mmlu_generative",
    "truthfulqa-multi_gen_en",
    "longbench",
}


def parse_scenario(value: str) -> tuple[int, int]:
    match = _SCENARIO_RE.fullmatch(value)
    if match is None:
        raise SiteDataError(f"invalid scenario: {value!r}")
    return int(match.group("prompt")), int(match.group("generation"))


def build_site_data(
    repo_root: Path,
    generated_at: str | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    registry = _load_registry(root)
    variants = {variant.key: variant for variant in registry.variants}

    speed_rows = _read_csv(root / "results" / "summary.csv")
    accuracy_rows = _read_csv(root / "results" / "eval_summary_primary.csv")
    mtplx_rows = _read_csv(root / "results" / "mtplx_speedups.csv")

    return {
        "generatedAt": generated_at or datetime.now(UTC).isoformat(),
        "benchVersion": _bench_version(speed_rows),
        "sourceCommit": source_commit if source_commit is not None else _source_commit(root),
        "hardware": {
            "machine": "Apple M5 Max",
            "memoryGb": 128,
            "os": "macOS 26.4",
        },
        "variants": [_variant_entry(variant) for variant in registry.variants],
        "accuracy": _accuracy_entries(accuracy_rows, variants),
        "speed": _speed_entries(speed_rows, registry),
        "mtplx": _mtplx_entries(mtplx_rows, variants),
        "caveats": _caveats(),
    }


def write_site_data(
    repo_root: Path,
    out_path: Path,
    generated_at: str | None = None,
) -> Path:
    data = build_site_data(repo_root=repo_root, generated_at=generated_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return out_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SiteDataError(f"required input file is missing: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_registry(repo_root: Path) -> Registry:
    path = repo_root / "models" / "registry.yaml"
    if not path.is_file():
        raise SiteDataError(f"required input file is missing: {path}")
    return load_registry(path)


def _source_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _bench_version(rows: Iterable[dict[str, str]]) -> str:
    candidates = [
        (row.get("ts", ""), index, row["bench_version"])
        for index, row in enumerate(rows)
        if row.get("bench_version", "")
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _variant_entry(variant: Variant) -> dict[str, Any]:
    return {
        "key": variant.key,
        "modelId": variant.model_id,
        "family": variant.family,
        "architecture": variant.architecture,
        "backend": variant.backend,
        "fmt": variant.fmt,
        "quant": variant.quant,
        "tier": variant.tier,
        "artifactType": variant.artifact_type,
        "approxSizeGb": variant.approx_size_gb,
        "paramsTotalB": variant.params_total_b,
        "paramsActiveB": variant.params_active_b,
        "generationMode": variant.generation_mode,
        "notes": variant.notes,
    }


def _accuracy_entries(
    rows: Iterable[dict[str, str]],
    variants: dict[str, Variant],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        variant_key = _required(row, "variant")
        if variant_key not in variants:
            raise SiteDataError(f"accuracy row references unknown variant: {variant_key}")
        variant = variants[variant_key]
        csv_model_id = row.get("model_id", "")
        if csv_model_id and csv_model_id != variant.model_id:
            raise SiteDataError(
                "accuracy row model_id does not match registry variant: "
                f"variant={variant_key}, csv_model_id={csv_model_id}, "
                f"registry_model_id={variant.model_id}"
            )
        task = _required(row, "task")
        metric = _required(row, "metric")
        entries.append(
            {
                "variant": variant_key,
                "modelId": variant.model_id,
                "dim": _required(row, "dim"),
                "task": task,
                "subtask": _required(row, "subtask"),
                "metric": metric,
                "value": _required_float(row, "value"),
                "stderr": _optional_float(row, "stderr"),
                "runId": _required(row, "run_id"),
                "timestamp": _required(row, "ts"),
                "confidence": _accuracy_confidence(task, metric),
                "caveats": _accuracy_caveats(task, metric),
            }
        )
    return entries


def _speed_entries(rows: Iterable[dict[str, str]], registry: Registry) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario = _required(row, "scenario")
        parse_scenario(scenario)
        variant = _speed_variant(row, registry)
        grouped[(variant.key, scenario)].append(row | {"variant_key": variant.key})

    entries: list[dict[str, Any]] = []
    for (variant_key, scenario), group in sorted(grouped.items()):
        variant = registry.variant(variant_key)
        prompt_tokens, generation_tokens = parse_scenario(scenario)
        timestamps = [_required(row, "ts") for row in group]
        entries.append(
            {
                "variant": variant_key,
                "modelId": variant.model_id,
                "scenario": scenario,
                "promptTokens": prompt_tokens,
                "generationTokens": generation_tokens,
                "ppTpsMean": _mean(_required_float(row, "pp_tps") for row in group),
                "tgTpsMean": _mean(_required_float(row, "tg_tps") for row in group),
                "peakMemGbMean": _mean(_required_float(row, "peak_mem_gb") for row in group),
                "wallSecondsMean": _mean(_required_float(row, "wall_s") for row in group),
                "runs": len(group),
                "runIndices": [_required_int(row, "run_idx") for row in group],
                "firstMeasuredAt": min(timestamps),
                "lastMeasuredAt": max(timestamps),
                "benchVersion": _bench_version(group),
                "backend": variant.backend,
                "fmt": variant.fmt,
                "quant": variant.quant,
                "tier": variant.tier,
                "ttftMs": None,
                "itlMs": None,
                "confidence": "measured",
                "caveats": ["latency-not-measured"],
            }
        )
    return entries


def _mtplx_entries(
    rows: Iterable[dict[str, str]],
    variants: dict[str, Variant],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        scenario = _required(row, "scenario")
        parse_scenario(scenario)
        mtp_variant = _required(row, "mtp_variant")
        ar_variant = _required(row, "ar_variant")
        for variant_key in (mtp_variant, ar_variant):
            if variant_key not in variants:
                raise SiteDataError(f"MTPLX row references unknown variant: {variant_key}")
        entries.append(
            {
                "pairKey": _required(row, "pair_key"),
                "scenario": scenario,
                "mtpVariant": mtp_variant,
                "arVariant": ar_variant,
                "mtpRuns": _required_int(row, "mtp_runs"),
                "arRuns": _required_int(row, "ar_runs"),
                "mtpTgTpsMean": _required_float(row, "mtp_tg_tps_mean"),
                "arTgTpsMean": _required_float(row, "ar_tg_tps_mean"),
                "speedup": _required_float(row, "speedup"),
                "mtpPeakMemGbMean": _required_float(row, "mtp_peak_mem_gb_mean"),
                "arPeakMemGbMean": _required_float(row, "ar_peak_mem_gb_mean"),
                "verifyMsPerCallMean": _required_float(row, "verify_ms_per_call_mean"),
                "acceptance": {
                    "d1": _required_float(row, "accept_d1_mean"),
                    "d2": _required_float(row, "accept_d2_mean"),
                    "d3": _required_float(row, "accept_d3_mean"),
                },
                "confidence": "measured",
                "caveats": [],
            }
        )
    return entries


def _speed_variant(row: dict[str, str], registry: Registry) -> Variant:
    variant_key = row.get("variant_key", "")
    if variant_key:
        try:
            return registry.variant(variant_key)
        except KeyError as exc:
            raise SiteDataError(f"speed row references unknown variant: {variant_key}") from exc

    candidates = [
        variant
        for variant in registry.variants
        if variant.model_id == row.get("model_id", "")
        and variant.fmt == row.get("fmt", "")
        and variant.backend == row.get("backend", "")
        and variant.quant == row.get("quant", "")
        and variant.tier == row.get("tier", "")
        and (
            not variant.generation_mode
            or variant.generation_mode == row.get("generation_mode", "")
        )
    ]
    if len(candidates) != 1:
        raise SiteDataError(
            "speed row has no unique registry variant: "
            f"model_id={row.get('model_id')!r}, fmt={row.get('fmt')!r}, "
            f"backend={row.get('backend')!r}, quant={row.get('quant')!r}, "
            f"tier={row.get('tier')!r}"
        )
    return candidates[0]


def _accuracy_confidence(task: str, metric: str) -> str:
    if task in _DIRECTIONAL_TASKS:
        return "directional"
    if "strict-match" in metric or "get_response" in metric:
        return "directional"
    return "measured"


def _accuracy_caveats(task: str, metric: str) -> list[str]:
    if _accuracy_confidence(task, metric) == "directional":
        return ["generative-exact-match"]
    return []


def _caveats() -> list[dict[str, str]]:
    return [
        {
            "id": "latency-not-measured",
            "status": "unavailable",
        },
        {
            "id": "generative-exact-match",
            "status": "directional",
        },
    ]


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise SiteDataError(f"required value is missing: {key}")
    return value


def _required_float(row: dict[str, str], key: str) -> float:
    return _parse_float(_required(row, key), key)


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return _parse_float(value, key)


def _parse_float(value: str, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SiteDataError(f"invalid numeric value for {key}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise SiteDataError(f"non-finite numeric value for {key}: {value!r}")
    return parsed


def _required_int(row: dict[str, str], key: str) -> int:
    value = _required(row, key)
    try:
        return int(value)
    except ValueError as exc:
        raise SiteDataError(f"invalid integer value for {key}: {value!r}") from exc


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise SiteDataError("cannot average an empty set")
    return sum(items) / len(items)
