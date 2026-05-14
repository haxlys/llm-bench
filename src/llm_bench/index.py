"""Build results/index.json — one-stop status snapshot for the dashboard.

Schema:
{
  "generated_at": "<iso ts>",
  "bench_version": "0.3",
  "registry_path": "models/registry.yaml",
  "variants": [
    {
      "key": "26B-MoE-mlx-8bit",
      "model_id": "gemma-4-26B-A4B-it",
      "fmt": "mlx", "quant": "MLX-8bit", "tier": "8bit",
      "family": "gemma", "architecture": "moe",
      "approx_size_gb": 26,
      "exists_locally": true,
      "speed": {
        "scenarios_measured": 8,
        "scenarios_total":    8,
        "last_measured":      "2026-04-28T14:18:00Z",
        "bench_versions":     ["0.3"]
      },
      "evals": {
        "tasks_measured":      5,  # supported measured tasks only
        "tasks_supported":     12, # full default suite + non-optional external tasks
        "last_measured":       "2026-04-28T08:13:00Z",
        "tasks":               ["mmlu_generative", "gsm8k_cot_zeroshot", ...],
        "extra_tasks":          ["older_smoke_task"]
      }
    },
    ...
  ],
  "totals": { "variants": 6, "speed_complete": 6, "evals_started": 1 }
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_bench import BENCH_VERSION, N_RUNS_REQUIRED
from llm_bench.evals.suites import (
    capabilities_for_backend,
    external_suite,
    external_supports_capabilities,
    full_suite,
    supports_capabilities,
    task_confidence,
    task_lane,
)
from llm_bench.manifest import eval_manifest, speed_manifest
from llm_bench.registry import get_registry, is_speed_only_variant
from llm_bench.scenarios import default_scenarios

ROOT = Path(__file__).resolve().parent.parent.parent


def build_index() -> dict:
    registry = get_registry()
    raw_dir = ROOT / "results" / "raw"
    eval_dir = ROOT / "results" / "eval_scores"

    speed_m = speed_manifest(raw_dir)
    eval_m = eval_manifest(eval_dir)
    scenarios = default_scenarios()
    full_tasks = full_suite()
    external_tasks = external_suite()
    task_catalog = _eval_task_catalog(full_tasks, external_tasks)

    variant_entries = []
    for v in registry.variants:
        speed_only = is_speed_only_variant(v)
        caps = getattr(
            v,
            "capabilities",
            capabilities_for_backend(getattr(v, "backend", v.fmt)),
        )
        s_measured = sum(
            1 for sc in scenarios
            if speed_m.counts.get((v.key, sc.name, BENCH_VERSION), 0) >= N_RUNS_REQUIRED
        )
        if speed_only:
            e_supported = set()
        else:
            e_supported = {t for _, t in full_tasks if supports_capabilities(t, caps)}
            e_supported.update(
                t for _, t, runner in external_tasks
                if _external_task_is_primary(t)
                and external_supports_capabilities(t, runner, caps)
            )
        e_measured_all = {t for (k, t) in eval_m.measured if k == v.key}
        e_measured_tasks = sorted(e_measured_all & e_supported)
        e_catalog_tasks = {entry["task"] for entry in task_catalog}
        e_extra_tasks = sorted(e_measured_all - e_catalog_tasks)
        coverage = [
            _coverage_row(entry, caps, e_measured_all, speed_only=speed_only)
            for entry in task_catalog
        ]
        coverage_summary = _coverage_summary(coverage)
        variant_entries.append({
            "key": v.key,
            "model_id": v.model_id,
            "fmt": v.fmt,
            "backend": getattr(v, "backend", v.fmt),
            "artifact_type": getattr(v, "artifact_type", ""),
            "capabilities": sorted(caps),
            "quant": v.quant,
            "tier": v.tier,
            "family": v.family,
            "architecture": v.architecture,
            "approx_size_gb": v.approx_size_gb,
            "exists_locally": v.exists_locally(),
            "speed": {
                "scenarios_measured": s_measured,
                "scenarios_total": len(scenarios),
                "last_measured": speed_m.last_ts.get(v.key) or None,
            },
            "evals": {
                "tasks_measured": len(e_measured_tasks),
                "tasks_supported": len(e_supported),
                "tasks": e_measured_tasks,
                "coverage": coverage,
                "coverage_summary": coverage_summary,
                "extra_tasks": e_extra_tasks,
                "last_measured": eval_m.last_ts.get(v.key) or None,
            },
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bench_version": BENCH_VERSION,
        "registry_path": "models/registry.yaml",
        "variants": variant_entries,
        "totals": {
            "variants": len(registry.variants),
            "speed_complete": sum(
                1 for e in variant_entries
                if e["speed"]["scenarios_measured"] == e["speed"]["scenarios_total"]
            ),
            "evals_started": sum(
                1 for e in variant_entries if e["evals"]["tasks_measured"] > 0
            ),
            "evals_missing_required": sum(
                e["evals"]["coverage_summary"]["missing"]
                for e in variant_entries
            ),
        },
    }


def write_index(out_path: Path | None = None) -> Path:
    out = out_path or (ROOT / "results" / "index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_index(), indent=2, ensure_ascii=False))
    return out


def _external_task_is_primary(task: str) -> bool:
    return task_lane(task) == "primary"


def _eval_task_catalog(
    full_tasks: list[tuple[str, str]],
    external_tasks: list[tuple[str, str, str]],
) -> list[dict[str, str]]:
    rows = [
        {"dim": dim, "task": task, "runner": "lm-eval", "lane": task_lane(task)}
        for dim, task in full_tasks
    ]
    rows.extend(
        {"dim": dim, "task": task, "runner": runner, "lane": task_lane(task)}
        for dim, task, runner in external_tasks
    )
    rows.append(
        {
            "dim": "agentic_code",
            "task": "programbench",
            "runner": "programbench",
            "lane": "optional",
        }
    )
    return rows


def _coverage_row(
    entry: dict[str, str],
    capabilities: set[str] | frozenset[str],
    measured_tasks: set[str],
    *,
    speed_only: bool = False,
) -> dict:
    task = entry["task"]
    lane = "mtplx_speedup" if speed_only else entry["lane"]
    measured = task in measured_tasks
    supported = measured if speed_only else _catalog_entry_supported(entry, capabilities)
    confidence = task_confidence(task)
    status = _coverage_status(
        lane=lane,
        supported=supported,
        measured=measured,
        confidence=confidence,
        speed_only=speed_only,
    )
    return {
        "dim": entry["dim"],
        "task": task,
        "runner": entry["runner"],
        "lane": lane,
        "required": lane == "primary",
        "supported": supported,
        "measured": measured,
        "confidence": confidence,
        "status": status,
    }


def _catalog_entry_supported(
    entry: dict[str, str],
    capabilities: set[str] | frozenset[str],
) -> bool:
    runner = entry["runner"]
    task = entry["task"]
    if runner == "lm-eval":
        return supports_capabilities(task, capabilities)
    if runner == "programbench":
        return True
    return external_supports_capabilities(task, runner, capabilities)


def _coverage_status(
    lane: str,
    supported: bool,
    measured: bool,
    confidence: str,
    *,
    speed_only: bool = False,
) -> str:
    if measured:
        return confidence
    if speed_only:
        return "speed_only"
    if lane == "diagnostic":
        return "diagnostic" if supported else "unsupported"
    if lane == "optional":
        return "optional" if supported else "unsupported"
    return "missing" if supported else "unsupported"


def _coverage_summary(rows: list[dict]) -> dict[str, int]:
    return {
        "measured": sum(1 for row in rows if row["status"] == "measured"),
        "directional": sum(1 for row in rows if row["status"] == "directional"),
        "diagnostic": sum(1 for row in rows if row["status"] == "diagnostic"),
        "missing": sum(1 for row in rows if row["status"] == "missing"),
        "optional": sum(1 for row in rows if row["status"] == "optional"),
        "speed_only": sum(1 for row in rows if row["status"] == "speed_only"),
        "unsupported": sum(1 for row in rows if row["status"] == "unsupported"),
        "total": len(rows),
    }
