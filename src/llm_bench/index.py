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
        "tasks_measured":      5,
        "tasks_supported":     12,
        "last_measured":       "2026-04-28T08:13:00Z",
        "tasks":               ["mmlu_generative", "gsm8k_cot_zeroshot", ...]
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

from llm_bench import BENCH_VERSION
from llm_bench.evals.suites import full_suite, supports_fmt
from llm_bench.manifest import eval_manifest, speed_manifest
from llm_bench.registry import get_registry
from llm_bench.scenarios import default_scenarios

ROOT = Path(__file__).resolve().parent.parent.parent


def _last_speed_ts(raw_dir: Path, variant_key: str) -> str | None:
    """Most recent ts across raw JSONs for this variant."""
    if not raw_dir.exists():
        return None
    latest = None
    for p in raw_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("variant_key") == variant_key or (
            data.get("model_id") and data.get("fmt") and data.get("quant")
            and any(v.key == variant_key
                    and v.model_id == data["model_id"]
                    and v.fmt == data["fmt"]
                    and v.quant == data["quant"]
                    for v in get_registry().variants)
        ):
            ts = data.get("ts", "")
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest


def _last_eval_ts(eval_dir: Path, variant_key: str) -> str | None:
    if not eval_dir.exists():
        return None
    candidates = [d.name.split("_", 1)[0] for d in eval_dir.iterdir()
                  if d.is_dir() and f"_{variant_key}_" in d.name]
    return max(candidates) if candidates else None


def build_index() -> dict:
    registry = get_registry()
    raw_dir = ROOT / "results" / "raw"
    eval_dir = ROOT / "results" / "eval_scores"

    speed_counts = speed_manifest(raw_dir)
    eval_pairs = eval_manifest(eval_dir)
    scenarios = default_scenarios()
    full_tasks = full_suite()

    variant_entries = []
    for v in registry.variants:
        s_measured = sum(
            1 for sc in scenarios
            if speed_counts.get((v.key, sc.name, BENCH_VERSION), 0) >= 3
        )
        e_supported = [t for _, t in full_tasks if supports_fmt(t, v.fmt)]
        e_measured_tasks = sorted(
            {t for (k, t) in eval_pairs if k == v.key}
        )
        variant_entries.append({
            "key": v.key,
            "model_id": v.model_id,
            "fmt": v.fmt,
            "quant": v.quant,
            "tier": v.tier,
            "family": v.family,
            "architecture": v.architecture,
            "approx_size_gb": v.approx_size_gb,
            "exists_locally": v.exists_locally(),
            "speed": {
                "scenarios_measured": s_measured,
                "scenarios_total": len(scenarios),
                "last_measured": _last_speed_ts(raw_dir, v.key),
            },
            "evals": {
                "tasks_measured": len(e_measured_tasks),
                "tasks_supported": len(e_supported),
                "tasks": e_measured_tasks,
                "last_measured": _last_eval_ts(eval_dir, v.key),
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
