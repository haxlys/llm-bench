# Benchmark Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static-first public benchmark website for `llm-bench` using TanStack Start and Cloudflare Workers Static Assets.

**Architecture:** Existing benchmark artifacts remain the source of truth. A Python exporter converts CSV/YAML inputs into `site/src/data/benchmarks.json`, and a TanStack Start app prerenders report/explorer routes that hydrate over this static JSON. Cloudflare Workers serves the prerendered output with Workers Static Assets; runtime APIs are outside v1.

**Tech Stack:** Python 3.13, pytest, pandas-free stdlib CSV parsing for the exporter, TanStack Start 1.167.64, React 19.2.5, Vite 8.0.10, Cloudflare Vite plugin 1.36.0, Wrangler 4.88.0, Vitest 4.1.5, Testing Library React 16.3.2, Recharts 3.8.1, lucide-react 1.14.0, Playwright 1.59.1.

---

## Reference Checks

Cloudflare docs checked on 2026-05-06:

- Workers Static Assets: `https://developers.cloudflare.com/workers/static-assets/`
- TanStack Start on Workers: `https://developers.cloudflare.com/workers/framework-guides/web-apps/tanstack-start/`
- Wrangler configuration: `https://developers.cloudflare.com/workers/wrangler/configuration/`
- Workers routes/domains: `https://developers.cloudflare.com/workers/configuration/routing/`

npm versions checked on 2026-05-06:

- `@tanstack/react-start@1.167.64`
- `@tanstack/react-router@1.169.2`
- `@cloudflare/vite-plugin@1.36.0`
- `wrangler@4.88.0`
- `react@19.2.5`
- `react-dom@19.2.5`
- `vite@8.0.10`
- `typescript@6.0.3`
- `vitest@4.1.5`

## File Structure

Create and modify these files:

- Create `src/llm_bench/site_data.py`: build the static website data model from registry, speed CSV, accuracy CSV, and MTPLX CSV.
- Create `scripts/export_site_data.py`: command-line wrapper that writes `site/src/data/benchmarks.json`.
- Create `tests/test_site_data.py`: exporter unit tests using small temporary fixtures.
- Create `site/package.json`: Node package scripts and pinned frontend dependencies.
- Create `site/tsconfig.json`: TypeScript configuration with JSON imports.
- Create `site/vite.config.ts`: TanStack Start + Cloudflare Vite plugin with prerender enabled.
- Create `site/wrangler.jsonc`: Worker deployment configuration.
- Create `site/app/router.tsx`: router creation.
- Create `site/app/routes/__root.tsx`: global shell, navigation, and metadata.
- Create `site/app/routes/index.tsx`: summary route.
- Create `site/app/routes/accuracy.tsx`: accuracy explorer route.
- Create `site/app/routes/speed.tsx`: speed explorer route.
- Create `site/app/routes/methodology.tsx`: methodology route.
- Create `site/app/routes/data.tsx`: downloads/data route.
- Create `site/app/components/*.tsx`: focused visual components.
- Create `site/app/lib/benchmark-data.ts`: typed data access and derived summaries.
- Create `site/app/lib/format.ts`: formatting helpers.
- Create `site/app/styles.css`: global responsive report styling.
- Create `site/app/lib/format.test.ts`: Vitest tests for formatting and metric status labels.
- Create `site/app/lib/benchmark-data.test.ts`: Vitest tests for derived top-level findings.
- Create `site/e2e/smoke.spec.ts`: Playwright route and screenshot smoke checks.
- Modify `.gitignore`: ignore TanStack/Workers generated folders.
- Modify `README.md`: add website build/deploy commands.

## Task 1: Website Data Exporter

**Files:**

- Create: `src/llm_bench/site_data.py`
- Create: `scripts/export_site_data.py`
- Create: `tests/test_site_data.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write exporter tests**

Create `tests/test_site_data.py` with this content:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_bench.site_data import build_site_data, parse_scenario, write_site_data


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
defaults: {}
models:
  - id: test-model
    family: qwen
    architecture: dense
    params_total_b: 4
    params_active_b: 4
    variants:
      - key: test-variant
        fmt: gguf
        path: /tmp/test.gguf
        quant: Q8_0
        tier: 8bit
        approx_size_gb: 4.5
  - id: mtplx-model
    family: qwen
    architecture: dense
    params_total_b: 27
    params_active_b: 27
    variants:
      - key: mtplx-mtp
        fmt: mlx
        backend: mtplx
        generation_mode: mtp
        path: org/mtp
        quant: MLX-4bit
        tier: 4bit
      - key: mtplx-ar
        fmt: mlx
        backend: mtplx
        generation_mode: ar
        path: org/ar
        quant: MLX-4bit
        tier: 4bit
""".strip()
    )


def test_parse_scenario_extracts_prompt_and_generation_tokens() -> None:
    assert parse_scenario("p4096_g512") == (4096, 512)


def test_build_site_data_preserves_zero_and_marks_unmeasured_latency(tmp_path: Path) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_csv(
        tmp_path / "results" / "summary.csv",
        [
            {
                "ts": "2026-05-06T00:00:00+00:00",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "scenario": "p256_g128",
                "generation_mode": "ar",
                "n_prompt": 256,
                "n_gen": 128,
                "pp_tps": 100.0,
                "tg_tps": 50.0,
                "peak_mem_gb": 4.5,
                "wall_s": 2.0,
                "run_idx": 1,
                "bench_version": "0.3",
                "variant_key": "test-variant",
                "tier": "8bit",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "eval_summary_primary.csv",
        [
            {
                "variant": "test-variant",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "tier": "8bit",
                "family": "qwen",
                "architecture": "dense",
                "dim": "reasoning",
                "task": "gsm8k_cot_zeroshot",
                "run_id": "run-1",
                "ts": "20260506T000000Z",
                "subtask": "gsm8k_cot_zeroshot",
                "metric": "exact_match,flexible-extract",
                "value": 0.0,
                "stderr": 0.0,
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "mtplx_speedups.csv",
        [
            {
                "pair_key": "mtplx-pair",
                "scenario": "p256_g128",
                "mtp_variant": "mtplx-mtp",
                "ar_variant": "mtplx-ar",
                "mtp_runs": 3,
                "ar_runs": 3,
                "mtp_tg_tps_mean": 40.0,
                "ar_tg_tps_mean": 25.0,
                "speedup": 1.6,
                "mtp_peak_mem_gb_mean": 15.0,
                "ar_peak_mem_gb_mean": 15.1,
                "verify_ms_per_call_mean": 51.0,
                "accept_d1_mean": 0.8,
                "accept_d2_mean": 0.6,
                "accept_d3_mean": 0.4,
            }
        ],
    )

    data = build_site_data(
        repo_root=tmp_path,
        generated_at="2026-05-06T12:00:00+00:00",
        source_commit="abc1234",
    )

    assert data["benchVersion"] == "0.3"
    assert data["sourceCommit"] == "abc1234"
    assert data["accuracy"][0]["value"] == 0.0
    assert data["accuracy"][0]["confidence"] == "directional"
    assert data["speed"][0]["ppTpsMean"] == 100.0
    assert data["speed"][0]["tgTpsMean"] == 50.0
    assert data["speed"][0]["ttftMs"] is None
    assert data["speed"][0]["itlMs"] is None
    assert data["mtplx"][0]["speedup"] == 1.6


def test_write_site_data_outputs_stable_json(tmp_path: Path) -> None:
    _write_registry(tmp_path / "models" / "registry.yaml")
    _write_csv(
        tmp_path / "results" / "summary.csv",
        [
            {
                "ts": "2026-05-06T00:00:00+00:00",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "scenario": "p256_g128",
                "generation_mode": "ar",
                "n_prompt": 256,
                "n_gen": 128,
                "pp_tps": 100.0,
                "tg_tps": 50.0,
                "peak_mem_gb": 4.5,
                "wall_s": 2.0,
                "run_idx": 1,
                "bench_version": "0.3",
                "variant_key": "test-variant",
                "tier": "8bit",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "eval_summary_primary.csv",
        [
            {
                "variant": "test-variant",
                "model_id": "test-model",
                "fmt": "gguf",
                "backend": "gguf",
                "artifact_type": "gguf_file",
                "quant": "Q8_0",
                "tier": "8bit",
                "family": "qwen",
                "architecture": "dense",
                "dim": "source_grounding",
                "task": "sourceqa",
                "run_id": "run-1",
                "ts": "20260506T000000Z",
                "subtask": "sourceqa",
                "metric": "acc,none",
                "value": 1.0,
                "stderr": "",
            }
        ],
    )
    _write_csv(
        tmp_path / "results" / "mtplx_speedups.csv",
        [
            {
                "pair_key": "mtplx-pair",
                "scenario": "p256_g128",
                "mtp_variant": "mtplx-mtp",
                "ar_variant": "mtplx-ar",
                "mtp_runs": 1,
                "ar_runs": 1,
                "mtp_tg_tps_mean": 40.0,
                "ar_tg_tps_mean": 25.0,
                "speedup": 1.6,
                "mtp_peak_mem_gb_mean": 15.0,
                "ar_peak_mem_gb_mean": 15.1,
                "verify_ms_per_call_mean": 51.0,
                "accept_d1_mean": 0.8,
                "accept_d2_mean": 0.6,
                "accept_d3_mean": 0.4,
            }
        ],
    )

    out = tmp_path / "site" / "src" / "data" / "benchmarks.json"
    write_site_data(tmp_path, out, generated_at="2026-05-06T12:00:00+00:00")

    parsed = json.loads(out.read_text())
    assert parsed["generatedAt"] == "2026-05-06T12:00:00+00:00"
    assert parsed["variants"][0]["key"] == "test-variant"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_site_data.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'llm_bench.site_data'`.

- [ ] **Step 3: Implement exporter module**

Create `src/llm_bench/site_data.py` with this content:

```python
"""Build static website data from committed benchmark artifacts."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_bench.registry import Registry, load_registry


class SiteDataError(ValueError):
    """Raised when website data cannot be generated from benchmark artifacts."""


def parse_scenario(value: str) -> tuple[int, int]:
    try:
        prompt, generation = value.split("_", 1)
        return int(prompt.removeprefix("p")), int(generation.removeprefix("g"))
    except (AttributeError, ValueError) as exc:
        raise SiteDataError(f"invalid scenario '{value}', expected p<N>_g<N>") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SiteDataError(f"required input file is missing: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SiteDataError(f"invalid numeric value for {key}: {raw!r}") from exc
    if not math.isfinite(value):
        raise SiteDataError(f"non-finite numeric value for {key}: {raw!r}")
    return value


def _int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "")
    try:
        return int(float(raw))
    except ValueError as exc:
        raise SiteDataError(f"invalid integer value for {key}: {raw!r}") from exc


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    return _float(row, key)


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


def _variant_key(row: dict[str, str]) -> str:
    key = row.get("variant_key") or row.get("variant")
    if key:
        return key
    raise SiteDataError(f"row has no variant key: {row}")


def _confidence_for_accuracy(task: str, metric: str) -> tuple[str, list[str]]:
    caveats: list[str] = []
    confidence = "measured"
    if task in {"gsm8k_cot_zeroshot", "hrm8k", "kmmlu_direct", "mmlu_generative"}:
        confidence = "directional"
        caveats.append("generative exact-match and extraction behavior can undercount correct answers")
    if task in {"truthfulqa-multi_gen_en", "longbench"}:
        confidence = "directional"
        caveats.append("legacy or limited-run metric kept for context")
    if "strict-match" in metric or "get_response" in metric:
        confidence = "directional"
        caveats.append("metric is sensitive to answer formatting")
    return confidence, sorted(set(caveats))


def _build_variants(registry: Registry) -> list[dict[str, Any]]:
    return [
        {
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
            "generationMode": variant.generation_mode or "ar",
            "notes": variant.notes,
        }
        for variant in registry.variants
    ]


def _build_accuracy(rows: list[dict[str, str]], registry: Registry) -> list[dict[str, Any]]:
    known = set(registry.variant_keys())
    output: list[dict[str, Any]] = []
    for row in rows:
        variant = _variant_key(row)
        if variant not in known:
            raise SiteDataError(f"accuracy row references unknown variant: {variant}")
        task = row["task"]
        metric = row["metric"]
        confidence, caveats = _confidence_for_accuracy(task, metric)
        output.append(
            {
                "variant": variant,
                "modelId": row["model_id"],
                "dim": row["dim"],
                "task": task,
                "subtask": row.get("subtask", ""),
                "metric": metric,
                "value": _float(row, "value"),
                "stderr": _optional_float(row, "stderr"),
                "runId": row["run_id"],
                "timestamp": row["ts"],
                "confidence": confidence,
                "caveats": caveats,
            }
        )
    return output


def _build_speed(rows: list[dict[str, str]], registry: Registry) -> list[dict[str, Any]]:
    known = set(registry.variant_keys())
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        variant = _variant_key(row)
        if variant not in known:
            raise SiteDataError(f"speed row references unknown variant: {variant}")
        groups[(variant, row["scenario"])].append(row)

    output: list[dict[str, Any]] = []
    for (variant, scenario), group in sorted(groups.items()):
        prompt_tokens, generation_tokens = parse_scenario(scenario)
        pp_values = [_float(row, "pp_tps") for row in group]
        tg_values = [_float(row, "tg_tps") for row in group]
        mem_values = [_float(row, "peak_mem_gb") for row in group]
        wall_values = [_float(row, "wall_s") for row in group]
        run_indices = [_int(row, "run_idx") for row in group if row.get("run_idx", "") != ""]
        first = group[0]
        output.append(
            {
                "variant": variant,
                "modelId": first["model_id"],
                "scenario": scenario,
                "promptTokens": prompt_tokens,
                "generationTokens": generation_tokens,
                "ppTpsMean": sum(pp_values) / len(pp_values),
                "tgTpsMean": sum(tg_values) / len(tg_values),
                "peakMemGbMean": sum(mem_values) / len(mem_values),
                "wallSecondsMean": sum(wall_values) / len(wall_values),
                "runs": len(group),
                "runIndices": run_indices,
                "firstMeasuredAt": min(row["ts"] for row in group),
                "lastMeasuredAt": max(row["ts"] for row in group),
                "benchVersion": first.get("bench_version", ""),
                "backend": first.get("backend", first.get("fmt", "")),
                "fmt": first.get("fmt", ""),
                "quant": first.get("quant", ""),
                "tier": first.get("tier", ""),
                "ttftMs": None,
                "itlMs": None,
                "confidence": "measured",
                "caveats": [],
            }
        )
    return output


def _build_mtplx(rows: list[dict[str, str]], registry: Registry) -> list[dict[str, Any]]:
    known = set(registry.variant_keys())
    output: list[dict[str, Any]] = []
    for row in rows:
        mtp_variant = row["mtp_variant"]
        ar_variant = row["ar_variant"]
        if mtp_variant not in known:
            raise SiteDataError(f"MTPLX row references unknown MTP variant: {mtp_variant}")
        if ar_variant not in known:
            raise SiteDataError(f"MTPLX row references unknown AR variant: {ar_variant}")
        output.append(
            {
                "pairKey": row["pair_key"],
                "scenario": row["scenario"],
                "mtpVariant": mtp_variant,
                "arVariant": ar_variant,
                "mtpRuns": _int(row, "mtp_runs"),
                "arRuns": _int(row, "ar_runs"),
                "mtpTgTpsMean": _float(row, "mtp_tg_tps_mean"),
                "arTgTpsMean": _float(row, "ar_tg_tps_mean"),
                "speedup": _float(row, "speedup"),
                "mtpPeakMemGbMean": _float(row, "mtp_peak_mem_gb_mean"),
                "arPeakMemGbMean": _float(row, "ar_peak_mem_gb_mean"),
                "verifyMsPerCallMean": _float(row, "verify_ms_per_call_mean"),
                "acceptance": {
                    "d1": _float(row, "accept_d1_mean"),
                    "d2": _float(row, "accept_d2_mean"),
                    "d3": _float(row, "accept_d3_mean"),
                },
                "confidence": "measured",
                "caveats": [],
            }
        )
    return output


def _bench_version(speed_rows: list[dict[str, str]]) -> str:
    versions = sorted({row.get("bench_version", "") for row in speed_rows if row.get("bench_version")})
    return versions[-1] if versions else ""


def build_site_data(
    repo_root: Path,
    generated_at: str | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    registry = load_registry(repo_root / "models" / "registry.yaml")
    speed_rows = _read_csv(repo_root / "results" / "summary.csv")
    accuracy_rows = _read_csv(repo_root / "results" / "eval_summary_primary.csv")
    mtplx_rows = _read_csv(repo_root / "results" / "mtplx_speedups.csv")
    generated = generated_at or datetime.now(tz=UTC).isoformat()
    return {
        "generatedAt": generated,
        "benchVersion": _bench_version(speed_rows),
        "sourceCommit": source_commit if source_commit is not None else _source_commit(repo_root),
        "hardware": {
            "machine": "Apple M5 Max",
            "memoryGb": 128,
            "os": "macOS 26.4",
        },
        "variants": _build_variants(registry),
        "accuracy": _build_accuracy(accuracy_rows, registry),
        "speed": _build_speed(speed_rows, registry),
        "mtplx": _build_mtplx(mtplx_rows, registry),
        "caveats": [
            {
                "key": "latency-not-measured",
                "status": "unavailable",
                "message": "TTFT and ITL are not directly measured in bench_version 0.3.",
            },
            {
                "key": "generative-exact-match",
                "status": "directional",
                "message": "Some generative exact-match tasks can undercount correct answers because extraction and formatting matter.",
            },
        ],
    }


def write_site_data(
    repo_root: Path,
    out_path: Path,
    generated_at: str | None = None,
) -> Path:
    data = build_site_data(repo_root, generated_at=generated_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return out_path
```

- [ ] **Step 4: Implement CLI wrapper**

Create `scripts/export_site_data.py` with this content:

```python
#!/usr/bin/env python3
"""Export benchmark website JSON from committed results."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.site_data import write_site_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("site/src/data/benchmarks.json"))
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else args.repo_root / args.out
    write_site_data(args.repo_root, out)
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Ignore generated frontend outputs**

Modify `.gitignore` by adding these lines near the existing cache/build ignores:

```gitignore
site/node_modules/
site/.vinxi/
site/.output/
site/.wrangler/
site/dist/
site/playwright-report/
site/test-results/
```

- [ ] **Step 6: Run exporter tests**

Run:

```bash
uv run pytest tests/test_site_data.py -q
```

Expected: PASS.

- [ ] **Step 7: Generate real site data**

Run:

```bash
uv run python scripts/export_site_data.py --out site/src/data/benchmarks.json
```

Expected: command prints `/Users/haxlys/projects/llm-bench/site/src/data/benchmarks.json`.

- [ ] **Step 8: Commit exporter**

Run:

```bash
git add .gitignore src/llm_bench/site_data.py scripts/export_site_data.py tests/test_site_data.py site/src/data/benchmarks.json
git commit -m "feat: export benchmark website data"
```

## Task 2: TanStack Start Cloudflare Scaffold

**Files:**

- Create: `site/package.json`
- Create: `site/tsconfig.json`
- Create: `site/vite.config.ts`
- Create: `site/wrangler.jsonc`
- Create: `site/app/router.tsx`
- Create: `site/app/routes/__root.tsx`
- Create: `site/app/styles.css`

- [ ] **Step 1: Create package manifest**

Create `site/package.json` with this content:

```json
{
  "name": "llm-bench-site",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "prebuild": "cd .. && uv run python scripts/export_site_data.py --out site/src/data/benchmarks.json",
    "build": "vite build",
    "preview": "wrangler dev",
    "deploy": "npm run build && wrangler deploy",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:e2e": "playwright test",
    "cf-typegen": "wrangler types"
  },
  "dependencies": {
    "@cloudflare/vite-plugin": "1.36.0",
    "@tanstack/react-router": "1.169.2",
    "@tanstack/react-start": "1.167.64",
    "@vitejs/plugin-react": "6.0.1",
    "lucide-react": "1.14.0",
    "react": "19.2.5",
    "react-dom": "19.2.5",
    "recharts": "3.8.1",
    "vite": "8.0.10"
  },
  "devDependencies": {
    "@playwright/test": "1.59.1",
    "@testing-library/react": "16.3.2",
    "@types/react": "19.2.14",
    "@types/react-dom": "19.2.3",
    "jsdom": "29.1.1",
    "typescript": "6.0.3",
    "vitest": "4.1.5",
    "wrangler": "4.88.0"
  }
}
```

- [ ] **Step 2: Install Node dependencies**

Run:

```bash
cd site && npm install
```

Expected: `site/package-lock.json` is created and no vulnerabilities block installation.

- [ ] **Step 3: Create TypeScript config**

Create `site/tsconfig.json` with this content:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "react-jsx",
    "types": ["vitest/globals"]
  },
  "include": ["app", "src/data/**/*.json", "vite.config.ts", "worker-configuration.d.ts"]
}
```

- [ ] **Step 4: Create Vite config**

Create `site/vite.config.ts` with this content:

```ts
import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    cloudflare({ viteEnvironment: { name: "ssr" } }),
    tanstackStart({
      prerender: {
        enabled: true,
      },
    }),
    react(),
  ],
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

- [ ] **Step 5: Create Wrangler config**

Create `site/wrangler.jsonc` with this content:

```jsonc
{
  "$schema": "./node_modules/wrangler/config-schema.json",
  "name": "llm-bench-site",
  "main": "@tanstack/react-start/server-entry",
  "compatibility_date": "2026-05-06",
  "compatibility_flags": ["nodejs_compat"],
  "observability": {
    "enabled": true
  }
}
```

- [ ] **Step 6: Create root shell**

Create `site/app/router.tsx` with this content:

```tsx
import { createRouter as createTanStackRouter } from "@tanstack/react-router";

import { routeTree } from "./routeTree.gen";

export function createRouter() {
  return createTanStackRouter({
    routeTree,
    scrollRestoration: true,
  });
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof createRouter>;
  }
}
```

Create `site/app/routes/__root.tsx` with this content:

```tsx
import { Link, Outlet, createRootRoute } from "@tanstack/react-router";
import { Database, Gauge, LineChart, ListChecks, Microscope } from "lucide-react";

import "../styles.css";

const navItems = [
  { to: "/", label: "Summary", icon: LineChart },
  { to: "/accuracy", label: "Accuracy", icon: ListChecks },
  { to: "/speed", label: "Speed", icon: Gauge },
  { to: "/methodology", label: "Methodology", icon: Microscope },
  { to: "/data", label: "Data", icon: Database },
] as const;

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <Link to="/" className="brand" aria-label="llm-bench summary">
          <span className="brand-mark">lb</span>
          <span>
            <strong>llm-bench</strong>
            <small>Apple Silicon local model benchmarks</small>
          </span>
        </Link>
        <nav className="site-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link key={item.to} to={item.to} className="nav-link">
                <Icon size={16} aria-hidden="true" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </header>
      <main className="site-main">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 7: Create base styles**

Create `site/app/styles.css` with this content:

```css
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --surface: #ffffff;
  --surface-muted: #f1f5f9;
  --border: #dbe3ef;
  --text: #101827;
  --muted: #5b677a;
  --accent: #0f766e;
  --accent-2: #2563eb;
  --warning: #b45309;
  --danger: #b91c1c;
  --radius: 8px;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
}

a {
  color: inherit;
}

.app-shell {
  min-height: 100vh;
}

.site-header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  border-bottom: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.94);
  padding: 12px 28px;
  backdrop-filter: blur(12px);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
}

.brand-mark {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  border-radius: var(--radius);
  background: var(--text);
  color: white;
  font-weight: 700;
}

.brand small {
  display: block;
  color: var(--muted);
  font-size: 12px;
}

.site-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.nav-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: var(--radius);
  color: var(--muted);
  padding: 8px 10px;
  text-decoration: none;
}

.nav-link.active {
  background: var(--surface-muted);
  color: var(--text);
}

.site-main {
  margin: 0 auto;
  max-width: 1180px;
  padding: 30px 24px 56px;
}

.section {
  margin-top: 28px;
}

.section-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.eyebrow {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.page-title {
  margin: 4px 0 8px;
  font-size: 38px;
  line-height: 1.08;
}

.lead {
  max-width: 780px;
  color: var(--muted);
  font-size: 17px;
  line-height: 1.6;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  padding: 16px;
}

.muted {
  color: var(--muted);
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.data-table th,
.data-table td {
  border-bottom: 1px solid var(--border);
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}

.data-table th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}

.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  background: var(--surface-muted);
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  padding: 3px 8px;
}

.badge.measured {
  background: #dcfce7;
  color: #166534;
}

.badge.directional {
  background: #fef3c7;
  color: var(--warning);
}

.badge.unavailable {
  background: #fee2e2;
  color: var(--danger);
}

.two-column {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
  gap: 16px;
}

@media (max-width: 900px) {
  .site-header {
    align-items: flex-start;
    flex-direction: column;
    padding: 12px 16px;
  }

  .site-main {
    padding: 24px 16px 42px;
  }

  .page-title {
    font-size: 30px;
  }

  .card-grid,
  .two-column {
    grid-template-columns: 1fr;
  }

  .data-table {
    display: block;
    overflow-x: auto;
  }
}
```

- [ ] **Step 8: Run typecheck to verify missing routes are the only blockers**

Run:

```bash
cd site && npm run typecheck
```

Expected: FAIL because `routeTree.gen` and route modules are not generated until routes are added and Vite/TanStack runs.

- [ ] **Step 9: Commit scaffold**

Run:

```bash
git add site/package.json site/package-lock.json site/tsconfig.json site/vite.config.ts site/wrangler.jsonc site/app/router.tsx site/app/routes/__root.tsx site/app/styles.css
git commit -m "feat: scaffold benchmark website"
```

## Task 3: Typed Site Data Utilities

**Files:**

- Create: `site/app/lib/benchmark-data.ts`
- Create: `site/app/lib/format.ts`
- Create: `site/app/lib/format.test.ts`
- Create: `site/app/lib/benchmark-data.test.ts`

- [ ] **Step 1: Write formatting tests**

Create `site/app/lib/format.test.ts` with this content:

```ts
import { describe, expect, it } from "vitest";

import { formatMetricValue, formatNumber, formatPercent, statusLabel } from "./format";

describe("format helpers", () => {
  it("keeps measured zero distinct from unavailable null", () => {
    expect(formatMetricValue(0, "percent")).toBe("0.0%");
    expect(formatMetricValue(null, "percent")).toBe("not measured");
  });

  it("formats percent and numeric values consistently", () => {
    expect(formatPercent(0.90853)).toBe("90.9%");
    expect(formatNumber(112.154)).toBe("112.15");
  });

  it("labels confidence statuses", () => {
    expect(statusLabel("measured")).toBe("measured");
    expect(statusLabel("directional")).toBe("directional");
    expect(statusLabel("unavailable")).toBe("not measured");
  });
});
```

- [ ] **Step 2: Implement format helpers**

Create `site/app/lib/format.ts` with this content:

```ts
export type MetricStatus = "measured" | "directional" | "unavailable";

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString("en-US", {
    maximumFractionDigits: 2,
    minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
  });
}

export function statusLabel(status: MetricStatus): string {
  if (status === "unavailable") {
    return "not measured";
  }
  return status;
}

export function formatMetricValue(
  value: number | null,
  kind: "percent" | "number" | "tokensPerSecond" | "memoryGb",
): string {
  if (value === null) {
    return "not measured";
  }
  if (kind === "percent") {
    return formatPercent(value);
  }
  if (kind === "tokensPerSecond") {
    return `${formatNumber(value)} tok/s`;
  }
  if (kind === "memoryGb") {
    return `${formatNumber(value)} GB`;
  }
  return formatNumber(value);
}
```

- [ ] **Step 3: Write derived data tests**

Create `site/app/lib/benchmark-data.test.ts` with this content:

```ts
import { describe, expect, it } from "vitest";

import type { BenchmarkData } from "./benchmark-data";
import { bestAccuracyRows, fastestSpeedRows, variantByKey } from "./benchmark-data";

const fixture: BenchmarkData = {
  generatedAt: "2026-05-06T12:00:00+00:00",
  benchVersion: "0.3",
  sourceCommit: "abc1234",
  hardware: { machine: "Apple M5 Max", memoryGb: 128, os: "macOS 26.4" },
  variants: [
    {
      key: "a",
      modelId: "model-a",
      family: "qwen",
      architecture: "moe",
      backend: "gguf",
      fmt: "gguf",
      quant: "Q4_K_M",
      tier: "4bit",
      artifactType: "gguf_file",
      approxSizeGb: 48.5,
      paramsTotalB: 48.5,
      paramsActiveB: 8,
      generationMode: "ar",
      notes: "",
    },
    {
      key: "b",
      modelId: "model-b",
      family: "gemma",
      architecture: "dense",
      backend: "gguf",
      fmt: "gguf",
      quant: "Q8_0",
      tier: "8bit",
      artifactType: "gguf_file",
      approxSizeGb: 8.2,
      paramsTotalB: 4,
      paramsActiveB: 4,
      generationMode: "ar",
      notes: "",
    },
  ],
  accuracy: [
    {
      variant: "a",
      modelId: "model-a",
      dim: "code",
      task: "humaneval",
      subtask: "humaneval",
      metric: "pass_at_1,base",
      value: 0.91,
      stderr: null,
      runId: "run-a",
      timestamp: "20260506T000000Z",
      confidence: "measured",
      caveats: [],
    },
    {
      variant: "b",
      modelId: "model-b",
      dim: "code",
      task: "humaneval",
      subtask: "humaneval",
      metric: "pass_at_1,base",
      value: 0.11,
      stderr: null,
      runId: "run-b",
      timestamp: "20260506T000000Z",
      confidence: "measured",
      caveats: [],
    },
  ],
  speed: [
    {
      variant: "a",
      modelId: "model-a",
      scenario: "p256_g128",
      promptTokens: 256,
      generationTokens: 128,
      ppTpsMean: 1500,
      tgTpsMean: 75,
      peakMemGbMean: 45,
      wallSecondsMean: 10,
      runs: 3,
      runIndices: [1, 2, 3],
      firstMeasuredAt: "2026-05-06T00:00:00+00:00",
      lastMeasuredAt: "2026-05-06T00:01:00+00:00",
      benchVersion: "0.3",
      backend: "gguf",
      fmt: "gguf",
      quant: "Q4_K_M",
      tier: "4bit",
      ttftMs: null,
      itlMs: null,
      confidence: "measured",
      caveats: [],
    },
    {
      variant: "b",
      modelId: "model-b",
      scenario: "p256_g128",
      promptTokens: 256,
      generationTokens: 128,
      ppTpsMean: 3000,
      tgTpsMean: 80,
      peakMemGbMean: 8,
      wallSecondsMean: 4,
      runs: 3,
      runIndices: [1, 2, 3],
      firstMeasuredAt: "2026-05-06T00:00:00+00:00",
      lastMeasuredAt: "2026-05-06T00:01:00+00:00",
      benchVersion: "0.3",
      backend: "gguf",
      fmt: "gguf",
      quant: "Q8_0",
      tier: "8bit",
      ttftMs: null,
      itlMs: null,
      confidence: "measured",
      caveats: [],
    },
  ],
  mtplx: [],
  caveats: [],
};

describe("benchmark data helpers", () => {
  it("indexes variants by key", () => {
    expect(variantByKey(fixture).get("a")?.modelId).toBe("model-a");
  });

  it("sorts top accuracy rows descending", () => {
    expect(bestAccuracyRows(fixture, "humaneval", 1)[0].variant).toBe("a");
  });

  it("sorts fastest speed rows descending by TG", () => {
    expect(fastestSpeedRows(fixture, "p256_g128", 1)[0].variant).toBe("b");
  });
});
```

- [ ] **Step 4: Implement data utilities**

Create `site/app/lib/benchmark-data.ts` with this content:

```ts
import data from "../../src/data/benchmarks.json";
import type { MetricStatus } from "./format";

export type Variant = {
  key: string;
  modelId: string;
  family: string;
  architecture: "dense" | "moe";
  backend: string;
  fmt: string;
  quant: string;
  tier: string;
  artifactType: string;
  approxSizeGb: number | null;
  paramsTotalB: number | null;
  paramsActiveB: number | null;
  generationMode: string;
  notes: string;
};

export type AccuracyRow = {
  variant: string;
  modelId: string;
  dim: string;
  task: string;
  subtask: string;
  metric: string;
  value: number;
  stderr: number | null;
  runId: string;
  timestamp: string;
  confidence: MetricStatus;
  caveats: string[];
};

export type SpeedRow = {
  variant: string;
  modelId: string;
  scenario: string;
  promptTokens: number;
  generationTokens: number;
  ppTpsMean: number;
  tgTpsMean: number;
  peakMemGbMean: number;
  wallSecondsMean: number;
  runs: number;
  runIndices: number[];
  firstMeasuredAt: string;
  lastMeasuredAt: string;
  benchVersion: string;
  backend: string;
  fmt: string;
  quant: string;
  tier: string;
  ttftMs: number | null;
  itlMs: number | null;
  confidence: MetricStatus;
  caveats: string[];
};

export type MtplxRow = {
  pairKey: string;
  scenario: string;
  mtpVariant: string;
  arVariant: string;
  mtpRuns: number;
  arRuns: number;
  mtpTgTpsMean: number;
  arTgTpsMean: number;
  speedup: number;
  mtpPeakMemGbMean: number;
  arPeakMemGbMean: number;
  verifyMsPerCallMean: number;
  acceptance: { d1: number; d2: number; d3: number };
  confidence: MetricStatus;
  caveats: string[];
};

export type BenchmarkData = {
  generatedAt: string;
  benchVersion: string;
  sourceCommit: string;
  hardware: { machine: string; memoryGb: number; os: string };
  variants: Variant[];
  accuracy: AccuracyRow[];
  speed: SpeedRow[];
  mtplx: MtplxRow[];
  caveats: { key: string; status: MetricStatus; message: string }[];
};

export const benchmarkData = data as BenchmarkData;

export function variantByKey(input: BenchmarkData): Map<string, Variant> {
  return new Map(input.variants.map((variant) => [variant.key, variant]));
}

export function bestAccuracyRows(
  input: BenchmarkData,
  task: string,
  limit: number,
): AccuracyRow[] {
  return input.accuracy
    .filter((row) => row.task === task)
    .toSorted((a, b) => b.value - a.value)
    .slice(0, limit);
}

export function fastestSpeedRows(
  input: BenchmarkData,
  scenario: string,
  limit: number,
): SpeedRow[] {
  return input.speed
    .filter((row) => row.scenario === scenario)
    .toSorted((a, b) => b.tgTpsMean - a.tgTpsMean)
    .slice(0, limit);
}

export function scenarios(input: BenchmarkData): string[] {
  return Array.from(new Set(input.speed.map((row) => row.scenario))).toSorted();
}

export function tasks(input: BenchmarkData): string[] {
  return Array.from(new Set(input.accuracy.map((row) => row.task))).toSorted();
}
```

- [ ] **Step 5: Run frontend unit tests**

Run:

```bash
cd site && npm test
```

Expected: PASS.

- [ ] **Step 6: Run typecheck**

Run:

```bash
cd site && npm run typecheck
```

Expected: still fails if routes are not complete; no failures should reference `format.ts` or `benchmark-data.ts`.

- [ ] **Step 7: Commit data utilities**

Run:

```bash
git add site/app/lib/format.ts site/app/lib/format.test.ts site/app/lib/benchmark-data.ts site/app/lib/benchmark-data.test.ts
git commit -m "feat: add typed benchmark site data helpers"
```

## Task 4: Summary Route And Shared Components

**Files:**

- Create: `site/app/components/Badge.tsx`
- Create: `site/app/components/FindingCard.tsx`
- Create: `site/app/components/CaveatCallout.tsx`
- Create: `site/app/components/AccuracyTable.tsx`
- Create: `site/app/components/SpeedBars.tsx`
- Create: `site/app/routes/index.tsx`
- Modify: `site/app/styles.css`

- [ ] **Step 1: Create shared badge and callout components**

Create `site/app/components/Badge.tsx` with this content:

```tsx
import type { MetricStatus } from "../lib/format";
import { statusLabel } from "../lib/format";

type BadgeProps = {
  status?: MetricStatus;
  children?: React.ReactNode;
};

export function Badge({ status, children }: BadgeProps) {
  const className = status ? `badge ${status}` : "badge";
  return <span className={className}>{children ?? statusLabel(status ?? "measured")}</span>;
}
```

Create `site/app/components/CaveatCallout.tsx` with this content:

```tsx
import { AlertTriangle } from "lucide-react";

type CaveatCalloutProps = {
  title: string;
  children: React.ReactNode;
};

export function CaveatCallout({ title, children }: CaveatCalloutProps) {
  return (
    <section className="panel caveat-callout">
      <AlertTriangle size={18} aria-hidden="true" />
      <div>
        <h2>{title}</h2>
        <div>{children}</div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Create finding card component**

Create `site/app/components/FindingCard.tsx` with this content:

```tsx
type FindingCardProps = {
  label: string;
  value: string;
  detail: string;
};

export function FindingCard({ label, value, detail }: FindingCardProps) {
  return (
    <article className="panel finding-card">
      <div className="eyebrow">{label}</div>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}
```

- [ ] **Step 3: Create summary table components**

Create `site/app/components/AccuracyTable.tsx` with this content:

```tsx
import type { AccuracyRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { Badge } from "./Badge";

type AccuracyTableProps = {
  rows: AccuracyRow[];
  variants: Map<string, Variant>;
};

export function AccuracyTable({ rows, variants }: AccuracyTableProps) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Task</th>
          <th>Metric</th>
          <th>Score</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const variant = variants.get(row.variant);
          return (
            <tr key={`${row.variant}-${row.task}-${row.metric}-${row.runId}`}>
              <td>
                <strong>{variant?.modelId ?? row.modelId}</strong>
                <div className="muted">{row.variant}</div>
              </td>
              <td>{row.task}</td>
              <td>{row.metric}</td>
              <td>{formatMetricValue(row.value, "percent")}</td>
              <td>
                <Badge status={row.confidence} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
```

Create `site/app/components/SpeedBars.tsx` with this content:

```tsx
import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";

type SpeedBarsProps = {
  rows: SpeedRow[];
  variants: Map<string, Variant>;
};

export function SpeedBars({ rows, variants }: SpeedBarsProps) {
  const max = Math.max(...rows.map((row) => row.tgTpsMean), 1);
  return (
    <div className="speed-bars">
      {rows.map((row) => {
        const variant = variants.get(row.variant);
        return (
          <div className="speed-row" key={`${row.variant}-${row.scenario}`}>
            <div className="speed-row-label">
              <strong>{variant?.modelId ?? row.modelId}</strong>
              <span>{formatMetricValue(row.tgTpsMean, "tokensPerSecond")}</span>
            </div>
            <div className="bar-track" aria-hidden="true">
              <div className="bar-fill" style={{ width: `${(row.tgTpsMean / max) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Create summary route**

Create `site/app/routes/index.tsx` with this content:

```tsx
import { createFileRoute } from "@tanstack/react-router";

import { AccuracyTable } from "../components/AccuracyTable";
import { CaveatCallout } from "../components/CaveatCallout";
import { FindingCard } from "../components/FindingCard";
import { SpeedBars } from "../components/SpeedBars";
import {
  benchmarkData,
  bestAccuracyRows,
  fastestSpeedRows,
  variantByKey,
} from "../lib/benchmark-data";

export const Route = createFileRoute("/")({
  component: SummaryRoute,
});

function SummaryRoute() {
  const variants = variantByKey(benchmarkData);
  const topAccuracy = bestAccuracyRows(benchmarkData, "humaneval", 5);
  const fastest = fastestSpeedRows(benchmarkData, "p256_g128", 5);
  const mtplxBest = benchmarkData.mtplx.toSorted((a, b) => b.speedup - a.speedup)[0];

  return (
    <>
      <section>
        <div className="eyebrow">
          {benchmarkData.hardware.machine}, {benchmarkData.hardware.memoryGb}GB unified memory
        </div>
        <h1 className="page-title">Local LLM benchmarks for speed, memory, and task quality</h1>
        <p className="lead">
          Public results from llm-bench, generated from committed CSV artifacts at commit{" "}
          <strong>{benchmarkData.sourceCommit || "unknown"}</strong>. The first version focuses on
          PP/TG throughput, peak memory, wall time, accuracy tables, and MTPLX speedups.
        </p>
      </section>

      <section className="section card-grid" aria-label="Key findings">
        <FindingCard
          label="Best code result"
          value="Qwen Coder Next Q4"
          detail="Leads HumanEval and MBPP among current published rows."
        />
        <FindingCard
          label="Fast 26B class"
          value="Gemma 26B MLX 4bit"
          detail="Strong generation throughput in the existing speed matrix."
        />
        <FindingCard
          label="Lightweight option"
          value="Qwen 3.5 4B Q8"
          detail="Small memory footprint with usable code scores for its size."
        />
        <FindingCard
          label="MTPLX speedup"
          value={mtplxBest ? `${mtplxBest.speedup.toFixed(2)}x` : "not available"}
          detail="Best measured MTP-on speedup over AR baseline in current data."
        />
      </section>

      <CaveatCallout title="Metric caveat">
        <p>
          TTFT and ITL are not directly measured in bench_version {benchmarkData.benchVersion}.
          Generative exact-match rows are marked directional when extraction or formatting can
          undercount correct answers.
        </p>
      </CaveatCallout>

      <section className="section two-column">
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">Accuracy snapshot</div>
              <h2>Top HumanEval rows</h2>
            </div>
          </div>
          <AccuracyTable rows={topAccuracy} variants={variants} />
        </div>
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">Token speed</div>
              <h2>Fastest p256_g128 TG</h2>
            </div>
          </div>
          <SpeedBars rows={fastest} variants={variants} />
        </div>
      </section>
    </>
  );
}
```

- [ ] **Step 5: Extend styles for summary components**

Append this to `site/app/styles.css`:

```css
.finding-card strong {
  display: block;
  margin-top: 8px;
  font-size: 20px;
}

.finding-card p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.5;
}

.caveat-callout {
  display: flex;
  gap: 12px;
  margin-top: 22px;
  border-color: #fcd34d;
  background: #fffbeb;
}

.caveat-callout h2 {
  margin: 0 0 4px;
  font-size: 16px;
}

.caveat-callout p {
  margin: 0;
  color: #7c2d12;
  line-height: 1.5;
}

.speed-bars {
  display: grid;
  gap: 14px;
}

.speed-row-label {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 13px;
}

.speed-row-label strong {
  color: var(--text);
}

.bar-track {
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--surface-muted);
}

.bar-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
}
```

- [ ] **Step 6: Run frontend tests and build**

Run:

```bash
cd site && npm test && npm run build
```

Expected: PASS and `.output/` is generated.

- [ ] **Step 7: Commit summary UI**

Run:

```bash
git add site/app/components site/app/routes/index.tsx site/app/styles.css
git commit -m "feat: add benchmark summary page"
```

## Task 5: Accuracy Explorer Route

**Files:**

- Create: `site/app/routes/accuracy.tsx`
- Modify: `site/app/components/AccuracyTable.tsx`
- Modify: `site/app/styles.css`

- [ ] **Step 1: Extend AccuracyTable with full columns**

Modify `site/app/components/AccuracyTable.tsx` so the table body also shows `dim`, `runId`, and caveat text. The row rendering block should become:

```tsx
<tr key={`${row.variant}-${row.task}-${row.metric}-${row.runId}`}>
  <td>
    <strong>{variant?.modelId ?? row.modelId}</strong>
    <div className="muted">{row.variant}</div>
  </td>
  <td>
    <strong>{row.task}</strong>
    <div className="muted">{row.dim}</div>
  </td>
  <td>{row.metric}</td>
  <td>{formatMetricValue(row.value, "percent")}</td>
  <td>
    <Badge status={row.confidence} />
    {row.caveats.length > 0 ? <div className="cell-note">{row.caveats.join("; ")}</div> : null}
  </td>
  <td className="run-id">{row.runId}</td>
</tr>
```

Add this `th` to the header:

```tsx
<th>Run</th>
```

- [ ] **Step 2: Create accuracy route**

Create `site/app/routes/accuracy.tsx` with this content:

```tsx
import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { AccuracyTable } from "../components/AccuracyTable";
import { benchmarkData, tasks, variantByKey } from "../lib/benchmark-data";

export const Route = createFileRoute("/accuracy")({
  component: AccuracyRoute,
});

function AccuracyRoute() {
  const allTasks = tasks(benchmarkData);
  const [task, setTask] = useState("all");
  const [family, setFamily] = useState("all");
  const variants = variantByKey(benchmarkData);
  const families = useMemo(
    () => Array.from(new Set(benchmarkData.variants.map((variant) => variant.family))).toSorted(),
    [],
  );
  const rows = benchmarkData.accuracy
    .filter((row) => task === "all" || row.task === task)
    .filter((row) => {
      const variant = variants.get(row.variant);
      return family === "all" || variant?.family === family;
    })
    .toSorted((a, b) => b.value - a.value);

  return (
    <>
      <section>
        <div className="eyebrow">Accuracy explorer</div>
        <h1 className="page-title">Task scores with metric and caveat context</h1>
        <p className="lead">
          Sort and filter the committed accuracy rows. Directional badges identify tasks where
          exact-match extraction, smoke limits, or formatting sensitivity affect interpretation.
        </p>
      </section>

      <section className="section panel filter-panel" aria-label="Accuracy filters">
        <label>
          Task
          <select value={task} onChange={(event) => setTask(event.target.value)}>
            <option value="all">All tasks</option>
            {allTasks.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          Family
          <select value={family} onChange={(event) => setFamily(event.target.value)}>
            <option value="all">All families</option>
            {families.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">{rows.length} rows</span>
      </section>

      <section className="section panel">
        <AccuracyTable rows={rows} variants={variants} />
      </section>
    </>
  );
}
```

- [ ] **Step 3: Add filter styles**

Append this to `site/app/styles.css`:

```css
.filter-panel {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 14px;
}

.filter-panel label {
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.filter-panel select {
  min-width: 190px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: white;
  color: var(--text);
  padding: 8px 10px;
  font: inherit;
  text-transform: none;
}

.cell-note {
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.4;
}

.run-id {
  max-width: 180px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
```

- [ ] **Step 4: Run frontend tests and build**

Run:

```bash
cd site && npm test && npm run build
```

Expected: PASS.

- [ ] **Step 5: Commit accuracy route**

Run:

```bash
git add site/app/routes/accuracy.tsx site/app/components/AccuracyTable.tsx site/app/styles.css
git commit -m "feat: add accuracy explorer"
```

## Task 6: Speed Explorer And MTPLX Panel

**Files:**

- Create: `site/app/components/SpeedScenarioTable.tsx`
- Create: `site/app/components/MtplxSpeedupPanel.tsx`
- Create: `site/app/routes/speed.tsx`
- Modify: `site/app/styles.css`

- [ ] **Step 1: Create speed table**

Create `site/app/components/SpeedScenarioTable.tsx` with this content:

```tsx
import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { Badge } from "./Badge";

type SpeedScenarioTableProps = {
  rows: SpeedRow[];
  variants: Map<string, Variant>;
};

export function SpeedScenarioTable({ rows, variants }: SpeedScenarioTableProps) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Scenario</th>
          <th>PP</th>
          <th>TG</th>
          <th>Memory</th>
          <th>Wall</th>
          <th>TTFT</th>
          <th>ITL</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const variant = variants.get(row.variant);
          return (
            <tr key={`${row.variant}-${row.scenario}`}>
              <td>
                <strong>{variant?.modelId ?? row.modelId}</strong>
                <div className="muted">{row.variant}</div>
              </td>
              <td>
                <strong>{row.scenario}</strong>
                <div className="muted">
                  {row.promptTokens} prompt / {row.generationTokens} gen
                </div>
              </td>
              <td>{formatMetricValue(row.ppTpsMean, "tokensPerSecond")}</td>
              <td>{formatMetricValue(row.tgTpsMean, "tokensPerSecond")}</td>
              <td>{formatMetricValue(row.peakMemGbMean, "memoryGb")}</td>
              <td>{formatMetricValue(row.wallSecondsMean, "number")}s</td>
              <td>{formatMetricValue(row.ttftMs, "number")}</td>
              <td>{formatMetricValue(row.itlMs, "number")}</td>
              <td>
                <Badge status={row.confidence} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Create MTPLX panel**

Create `site/app/components/MtplxSpeedupPanel.tsx` with this content:

```tsx
import type { MtplxRow } from "../lib/benchmark-data";
import { formatMetricValue, formatNumber } from "../lib/format";

type MtplxSpeedupPanelProps = {
  rows: MtplxRow[];
};

export function MtplxSpeedupPanel({ rows }: MtplxSpeedupPanelProps) {
  const sorted = rows.toSorted((a, b) => b.speedup - a.speedup);
  return (
    <div className="mtplx-grid">
      {sorted.map((row) => (
        <article className="panel" key={`${row.pairKey}-${row.scenario}`}>
          <div className="eyebrow">{row.scenario}</div>
          <h2>{formatNumber(row.speedup)}x speedup</h2>
          <p className="muted">
            MTP {formatMetricValue(row.mtpTgTpsMean, "tokensPerSecond")} vs AR{" "}
            {formatMetricValue(row.arTgTpsMean, "tokensPerSecond")}
          </p>
          <dl className="compact-dl">
            <div>
              <dt>accept d1</dt>
              <dd>{formatNumber(row.acceptance.d1)}</dd>
            </div>
            <div>
              <dt>accept d2</dt>
              <dd>{formatNumber(row.acceptance.d2)}</dd>
            </div>
            <div>
              <dt>accept d3</dt>
              <dd>{formatNumber(row.acceptance.d3)}</dd>
            </div>
          </dl>
        </article>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Create speed route**

Create `site/app/routes/speed.tsx` with this content:

```tsx
import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import { MtplxSpeedupPanel } from "../components/MtplxSpeedupPanel";
import { SpeedBars } from "../components/SpeedBars";
import { SpeedScenarioTable } from "../components/SpeedScenarioTable";
import { benchmarkData, scenarios, variantByKey } from "../lib/benchmark-data";

export const Route = createFileRoute("/speed")({
  component: SpeedRoute,
});

function SpeedRoute() {
  const allScenarios = scenarios(benchmarkData);
  const [scenario, setScenario] = useState(allScenarios[0] ?? "p256_g128");
  const variants = variantByKey(benchmarkData);
  const rows = benchmarkData.speed
    .filter((row) => row.scenario === scenario)
    .toSorted((a, b) => b.tgTpsMean - a.tgTpsMean);

  return (
    <>
      <section>
        <div className="eyebrow">Speed explorer</div>
        <h1 className="page-title">Token throughput, memory, and MTPLX speedups</h1>
        <p className="lead">
          Current artifacts measure PP tok/s, TG tok/s, peak memory, and wall time. TTFT and
          ITL are shown as not measured until the runner records them directly.
        </p>
      </section>

      <section className="section panel filter-panel" aria-label="Speed filters">
        <label>
          Scenario
          <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
            {allScenarios.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">{rows.length} model rows</span>
      </section>

      <section className="section two-column">
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">TG tok/s</div>
              <h2>Generation throughput</h2>
            </div>
          </div>
          <SpeedBars rows={rows.slice(0, 8)} variants={variants} />
        </div>
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">Unavailable metrics</div>
              <h2>TTFT and ITL</h2>
            </div>
          </div>
          <p className="muted">
            These fields are intentionally null in bench_version {benchmarkData.benchVersion}.
            The site does not estimate latency from aggregate throughput.
          </p>
        </div>
      </section>

      <section className="section panel">
        <SpeedScenarioTable rows={rows} variants={variants} />
      </section>

      <section className="section">
        <div className="section-header">
          <div>
            <div className="eyebrow">MTPLX</div>
            <h2>MTP-on versus AR baseline</h2>
          </div>
        </div>
        <MtplxSpeedupPanel rows={benchmarkData.mtplx} />
      </section>
    </>
  );
}
```

- [ ] **Step 4: Add MTPLX styles**

Append this to `site/app/styles.css`:

```css
.mtplx-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.mtplx-grid h2 {
  margin: 8px 0;
  font-size: 24px;
}

.compact-dl {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 12px 0 0;
}

.compact-dl div {
  border-radius: var(--radius);
  background: var(--surface-muted);
  padding: 8px;
}

.compact-dl dt {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
}

.compact-dl dd {
  margin: 2px 0 0;
  font-weight: 700;
}

@media (max-width: 900px) {
  .mtplx-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 5: Run frontend tests and build**

Run:

```bash
cd site && npm test && npm run build
```

Expected: PASS.

- [ ] **Step 6: Commit speed route**

Run:

```bash
git add site/app/components/SpeedScenarioTable.tsx site/app/components/MtplxSpeedupPanel.tsx site/app/routes/speed.tsx site/app/styles.css
git commit -m "feat: add speed explorer"
```

## Task 7: Methodology And Data Routes

**Files:**

- Create: `site/app/components/MetricGlossary.tsx`
- Create: `site/app/components/DownloadLinks.tsx`
- Create: `site/app/routes/methodology.tsx`
- Create: `site/app/routes/data.tsx`
- Modify: `site/app/styles.css`

- [ ] **Step 1: Create metric glossary**

Create `site/app/components/MetricGlossary.tsx` with this content:

```tsx
const metrics = [
  ["PP tok/s", "Prompt processing throughput reported by the benchmark runner."],
  ["TG tok/s", "Generation throughput reported by the benchmark runner."],
  ["Peak memory", "Peak process memory in GB, using the local benchmark measurement path."],
  ["Wall time", "End-to-end run duration including load and execution overhead."],
  ["TTFT", "Time to first token. Not measured in bench_version 0.3."],
  ["ITL", "Inter-token latency. Not measured in bench_version 0.3."],
] as const;

export function MetricGlossary() {
  return (
    <dl className="glossary">
      {metrics.map(([term, definition]) => (
        <div key={term}>
          <dt>{term}</dt>
          <dd>{definition}</dd>
        </div>
      ))}
    </dl>
  );
}
```

- [ ] **Step 2: Create download links component**

Create `site/app/components/DownloadLinks.tsx` with this content:

```tsx
const links = [
  { href: "/data/benchmarks.json", label: "Generated website JSON" },
  { href: "/data/summary.csv", label: "Speed summary CSV" },
  { href: "/data/eval_summary_primary.csv", label: "Accuracy primary CSV" },
  { href: "/data/mtplx_speedups.csv", label: "MTPLX speedups CSV" },
] as const;

export function DownloadLinks() {
  return (
    <ul className="download-list">
      {links.map((link) => (
        <li key={link.href}>
          <a href={link.href}>{link.label}</a>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 3: Create methodology route**

Create `site/app/routes/methodology.tsx` with this content:

```tsx
import { createFileRoute } from "@tanstack/react-router";

import { CaveatCallout } from "../components/CaveatCallout";
import { MetricGlossary } from "../components/MetricGlossary";
import { benchmarkData } from "../lib/benchmark-data";

export const Route = createFileRoute("/methodology")({
  component: MethodologyRoute,
});

function MethodologyRoute() {
  return (
    <>
      <section>
        <div className="eyebrow">Methodology</div>
        <h1 className="page-title">How to read these local benchmark results</h1>
        <p className="lead">
          Results are generated from committed artifacts for bench_version{" "}
          {benchmarkData.benchVersion}. The current hardware target is{" "}
          {benchmarkData.hardware.machine} with {benchmarkData.hardware.memoryGb}GB unified memory.
        </p>
      </section>

      <section className="section panel">
        <h2>Metric glossary</h2>
        <MetricGlossary />
      </section>

      <CaveatCallout title="Accuracy caveats">
        <p>
          Some generative exact-match tasks can undercount correct answers when output formatting
          differs from the extractor. The site marks those rows directional instead of hiding them.
        </p>
      </CaveatCallout>

      <section className="section panel">
        <h2>Scenario matrix</h2>
        <p className="muted">
          Speed rows use prompt/generation scenario keys such as p256_g128, p1024_g512,
          p4096_g128, and p8192_g512. Each row reports current committed aggregate values.
        </p>
      </section>
    </>
  );
}
```

- [ ] **Step 4: Create data route**

Create `site/app/routes/data.tsx` with this content:

```tsx
import { createFileRoute } from "@tanstack/react-router";

import { DownloadLinks } from "../components/DownloadLinks";
import { benchmarkData } from "../lib/benchmark-data";

export const Route = createFileRoute("/data")({
  component: DataRoute,
});

function DataRoute() {
  return (
    <>
      <section>
        <div className="eyebrow">Data</div>
        <h1 className="page-title">Download the benchmark artifacts behind the site</h1>
        <p className="lead">
          Generated at {benchmarkData.generatedAt}. Source commit{" "}
          <strong>{benchmarkData.sourceCommit || "unknown"}</strong>.
        </p>
      </section>

      <section className="section panel">
        <h2>Downloads</h2>
        <DownloadLinks />
      </section>
    </>
  );
}
```

- [ ] **Step 5: Copy CSV downloads into public data path**

Create `site/public/data/` and copy the committed CSVs there:

```bash
mkdir -p site/public/data
cp results/summary.csv site/public/data/summary.csv
cp results/eval_summary_primary.csv site/public/data/eval_summary_primary.csv
cp results/mtplx_speedups.csv site/public/data/mtplx_speedups.csv
cp site/src/data/benchmarks.json site/public/data/benchmarks.json
```

- [ ] **Step 6: Add glossary/download styles**

Append this to `site/app/styles.css`:

```css
.glossary {
  display: grid;
  gap: 10px;
  margin: 0;
}

.glossary div {
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
}

.glossary dt {
  font-weight: 700;
}

.glossary dd {
  margin: 4px 0 0;
  color: var(--muted);
  line-height: 1.5;
}

.download-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.download-list a {
  display: block;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  text-decoration: none;
}

.download-list a:hover {
  border-color: var(--accent);
}
```

- [ ] **Step 7: Run frontend tests and build**

Run:

```bash
cd site && npm test && npm run build
```

Expected: PASS.

- [ ] **Step 8: Commit methodology/data routes**

Run:

```bash
git add site/app/components/MetricGlossary.tsx site/app/components/DownloadLinks.tsx site/app/routes/methodology.tsx site/app/routes/data.tsx site/app/styles.css site/public/data
git commit -m "feat: add methodology and data pages"
```

## Task 8: Verification, Preview, And Documentation

**Files:**

- Create: `site/e2e/smoke.spec.ts`
- Create: `site/playwright.config.ts`
- Modify: `README.md`

- [ ] **Step 1: Create Playwright config**

Create `site/playwright.config.ts` with this content:

```ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});
```

- [ ] **Step 2: Create smoke test**

Create `site/e2e/smoke.spec.ts` with this content:

```ts
import { expect, test } from "@playwright/test";

const routes = ["/", "/accuracy", "/speed", "/methodology", "/data"] as const;

for (const route of routes) {
  test(`renders ${route}`, async ({ page }) => {
    await page.goto(route);
    await expect(page.locator("body")).toContainText("llm-bench");
    await expect(page.locator("main")).toBeVisible();
  });
}

test("marks TTFT and ITL as not measured", async ({ page }) => {
  await page.goto("/speed");
  await expect(page.locator("main")).toContainText("TTFT");
  await expect(page.locator("main")).toContainText("not measured");
});
```

- [ ] **Step 3: Run full frontend verification**

Run:

```bash
cd site && npm test && npm run typecheck && npm run build && npm run test:e2e
```

Expected: all commands PASS.

- [ ] **Step 4: Preview Workers build locally**

Run:

```bash
cd site && npm run build && npm run preview
```

Expected: Wrangler prints a local URL and the Summary, Accuracy, Speed, Methodology, and Data routes render from the preview server. Stop the preview server after checking.

- [ ] **Step 5: Update README**

Append this section to `README.md`:

````markdown
## Public benchmark website

The static public benchmark site lives in `site/` and is built from committed result artifacts.

```bash
# Regenerate static website data
uv run python scripts/export_site_data.py --out site/src/data/benchmarks.json

# Install frontend dependencies
cd site && npm install

# Local development
npm run dev

# Production build
npm run build

# Cloudflare Workers preview
npm run preview

# Deploy to Cloudflare Workers
npm run deploy
```

The site uses Cloudflare Workers Static Assets through the Cloudflare Vite plugin and TanStack Start static prerendering. `TTFT` and `ITL` are displayed as not measured until the benchmark runner records them directly.
````

- [ ] **Step 6: Run repository verification**

Run:

```bash
uv run pytest -q
cd site && npm test && npm run typecheck && npm run build
```

Expected: all commands PASS.

- [ ] **Step 7: Commit verification docs**

Run:

```bash
git add README.md site/e2e/smoke.spec.ts site/playwright.config.ts
git commit -m "test: verify benchmark website"
```

## Task 9: Deployment Preparation

**Files:**

- Modify: `site/wrangler.jsonc`

- [ ] **Step 1: Choose Worker name and route**

Use `llm-bench-site` as the Worker name for v1 unless the Cloudflare dashboard project already uses another name. If a custom domain is ready, add a route or custom domain through Cloudflare dashboard or Wrangler after the first successful preview.

- [ ] **Step 2: Run production deploy dry check**

Run:

```bash
cd site && npm run build && npx wrangler deploy --dry-run
```

Expected: Wrangler validates the Worker bundle and assets without publishing.

- [ ] **Step 3: Deploy**

Run:

```bash
cd site && npm run deploy
```

Expected: Wrangler publishes the Worker and prints a deployment URL.

- [ ] **Step 4: Verify deployed site**

Open the deployed URL and verify:

- Summary route renders.
- Accuracy filters work.
- Speed scenario selector works.
- TTFT and ITL show as not measured.
- Data downloads return JSON/CSV files.

- [ ] **Step 5: Commit any final deployment config**

If `site/wrangler.jsonc` changed for route/custom-domain configuration, run:

```bash
git add site/wrangler.jsonc
git commit -m "chore: configure benchmark website deployment"
```

If no files changed, do not create an empty commit.

## Self-Review Notes

Spec coverage:

- Hybrid public report plus explorer: Tasks 4, 5, 6, and 7.
- Workers Static Assets and TanStack Start prerender: Task 2 and Task 9.
- CSV-to-JSON data pipeline: Task 1.
- Accuracy table: Tasks 4 and 5.
- Speed PP/TG/memory/wall time: Task 6.
- TTFT/ITL unavailable treatment: Tasks 1, 3, 6, and 8.
- Caveat/confidence model: Tasks 1, 3, 4, and 5.
- Methodology and downloads: Task 7.
- Verification and preview: Task 8.

Gap scan:

- This plan does not use open-ended implementation gaps.
- Every command includes an expected result.
- Every created file has a concrete target path and concrete content or an exact edit target.

Type consistency:

- Frontend uses `MetricStatus`, `BenchmarkData`, `AccuracyRow`, `SpeedRow`, and `MtplxRow` consistently.
- Exporter keys use camelCase JSON names matching TypeScript types.
- `ttftMs` and `itlMs` are `null` in Python output and `number | null` in TypeScript.
