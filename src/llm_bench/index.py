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
)
from llm_bench.manifest import eval_manifest, speed_manifest
from llm_bench.registry import get_registry
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

    variant_entries = []
    for v in registry.variants:
        caps = getattr(
            v,
            "capabilities",
            capabilities_for_backend(getattr(v, "backend", v.fmt)),
        )
        s_measured = sum(
            1 for sc in scenarios
            if speed_m.counts.get((v.key, sc.name, BENCH_VERSION), 0) >= N_RUNS_REQUIRED
        )
        e_supported = {t for _, t in full_tasks if supports_capabilities(t, caps)}
        e_supported.update(
            t for _, t, runner in external_suite()
            if runner != "bfcl" and external_supports_capabilities(t, runner, caps)
        )
        e_measured_all = {t for (k, t) in eval_m.measured if k == v.key}
        e_measured_tasks = sorted(e_measured_all & e_supported)
        e_extra_tasks = sorted(e_measured_all - e_supported)
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
        },
    }


def write_index(out_path: Path | None = None) -> Path:
    out = out_path or (ROOT / "results" / "index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_index(), indent=2, ensure_ascii=False))
    return out
