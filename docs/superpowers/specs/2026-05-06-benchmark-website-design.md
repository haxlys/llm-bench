# Benchmark Website Design

Date: 2026-05-06

## Purpose

Build a public benchmark website for `llm-bench` that presents the current local LLM benchmark results as both a credible report and an interactive comparison tool.

The first version should answer three reader questions quickly:

1. Which tested model looks strongest overall?
2. Which models are fastest on the current machine?
3. Which results are measured directly, directional, or not measured yet?

The site should be suitable for public sharing and later expansion, but v1 should stay static-first and auditable from committed benchmark artifacts.

## Audience

The primary audience is technical readers who may want to cite or compare the results: local LLM users, benchmarkers, engineers choosing a model variant, and future contributors to this repo.

The site should not feel like a marketing landing page. It should feel like a compact public benchmark report with clear tables, careful caveats, and fast drill-down controls.

## Platform Decision

Use Cloudflare Workers with Workers Static Assets and TanStack Start static prerendering.

This follows the current Cloudflare guidance:

- Workers Static Assets is the modern path for serving static assets together with Worker code.
- Workers Sites is deprecated for new projects in Wrangler v4.
- Cloudflare's TanStack Start guide documents static prerendering through `@tanstack/react-start` v1.138.0 or newer.
- Production deployment should use a custom domain or Worker route rather than relying on a `workers.dev` subdomain.

The v1 site should not require runtime data APIs. Worker code is reserved for routing, headers, and future API expansion.

## Information Architecture

The site uses a hybrid report-plus-explorer structure.

Routes:

- `/`: summary report with key findings, top accuracy rows, speed preview, and caveats.
- `/accuracy`: sortable/filterable accuracy explorer.
- `/speed`: token throughput, memory, scenario, and MTPLX speedup explorer.
- `/methodology`: hardware, protocol, scenario matrix, benchmark version, and known limitations.
- `/data`: generated JSON, original CSV links, source metadata, and update timestamp.

The first viewport should show:

- Hardware and date context.
- Best overall result.
- Fastest practical result.
- Lightweight result.
- MTPLX speculative decoding speedup.
- A visible caveat that TTFT and ITL are not measured in the current data.

## Data Flow

Benchmark CSVs remain the source of truth:

- `results/summary.csv`
- `results/eval_summary_primary.csv`
- `results/mtplx_speedups.csv`
- `models/registry.yaml`

A build-time export script should normalize these inputs into a single site data file:

- `site/src/data/benchmarks.json`

Flow:

```text
Benchmark CSVs
  -> export_site_data.py
  -> site/src/data/benchmarks.json
  -> TanStack Start prerender
  -> Workers Static Assets deployment
  -> custom domain
```

The export script should fail clearly when required inputs are missing or malformed. It should not silently drop unknown metrics.

## Data Schema

The generated JSON should include:

- `generatedAt`
- `benchVersion`
- `sourceCommit`
- `hardware`
- `variants`
- `accuracy`
- `speed`
- `mtplx`
- `caveats`

Variant rows should include:

- `key`
- `modelId`
- `family`
- `architecture`
- `backend`
- `fmt`
- `quant`
- `tier`
- `approxSizeGb`

Accuracy rows should include:

- `variant`
- `task`
- `dim`
- `metric`
- `value`
- `runId`
- `timestamp`
- `confidence`
- `caveats`

Speed rows should include:

- `variant`
- `scenario`
- `promptTokens`
- `generationTokens`
- `ppTpsMean`
- `tgTpsMean`
- `peakMemGbMean`
- `wallSecondsMean`
- `ttftMs`
- `itlMs`
- `runs`

For v1, `ttftMs` and `itlMs` must be `null` because the current artifacts do not directly measure them.

MTPLX rows should include:

- `pairKey`
- `scenario`
- `mtpVariant`
- `arVariant`
- `mtpTgTpsMean`
- `arTgTpsMean`
- `speedup`
- `mtpPeakMemGbMean`
- `arPeakMemGbMean`
- `acceptance`

## Metric Confidence

Every displayed metric should have a confidence status.

Statuses:

- `measured`: directly present in the input artifacts with a metric name and run id.
- `directional`: measured, but interpretation requires a caveat such as exact-match extraction issues, smoke limits, or small sample size.
- `unavailable`: requested or expected metric is not measured in the current benchmark version.

Rules:

- Display `0` only for a real measured zero.
- Display `null` values as `not measured`, `not available`, or an equivalent explicit label.
- Do not infer TTFT or ITL from aggregate throughput.
- Do not compute a blended leaderboard score until the weighting policy is explicit.

## Components

Core components:

- `FindingCard`: top-level conclusion cards.
- `BenchmarkBadge`: backend, quantization, suite, and confidence labels.
- `AccuracyTable`: sortable table for task-level accuracy.
- `SpeedScenarioTable`: scenario-level speed and memory table.
- `SpeedChart`: PP/TG bar chart with scenario selection.
- `MtplxSpeedupPanel`: MTP-on versus AR baseline comparison.
- `MetricGlossary`: PP, TG, TTFT, ITL, wall time, memory, and pass@1 definitions.
- `CaveatCallout`: visible caveat blocks near affected scores.
- `DownloadLinks`: generated JSON and original CSV access.

Tables must work well on desktop and mobile. On small screens, prefer compact cards and metric selectors over forcing an unreadable wide table.

## Visual Direction

Use a restrained technical report style:

- Dense but legible information hierarchy.
- Neutral background, high-contrast text, and limited accent colors.
- Tables and charts optimized for scanning.
- No marketing hero, decorative gradient background, or purely illustrative imagery.

The first screen should make the subject obvious: local LLM benchmarks on Apple M5 Max using `llm-bench`.

## Error Handling

Build-time validation should catch:

- Missing required CSVs.
- Missing registry metadata for a displayed variant.
- Unknown metric names that are not mapped.
- Empty numeric fields where a number is required.
- NaN or infinite numeric values.
- Duplicate primary rows for the same variant/task/metric unless the export script intentionally chooses the latest run.

Runtime UI should handle:

- Empty filter results.
- Missing optional metrics.
- Unavailable TTFT/ITL fields.
- Long model names.
- Mobile layout constraints.

## Testing Strategy

Data tests:

- CSV-to-JSON export produces valid schema.
- TTFT and ITL remain `null` unless directly measured.
- `0` and `null` are displayed differently.
- Confidence status is assigned as expected.
- MTPLX speedup rows preserve AR/MTP pairing.

UI tests:

- Accuracy table sorts and filters correctly.
- Speed scenario selector changes the displayed rows/charts.
- Caveat badges appear for directional and unavailable metrics.
- Mobile layout does not overlap text or controls.

Build and deployment tests:

- `npm run build` completes.
- TanStack Start prerender emits all planned routes.
- Workers preview serves static assets correctly.
- Desktop and mobile screenshots are nonblank and readable.

## Non-Goals For v1

- User accounts.
- Live benchmark execution from the website.
- Runtime uploads.
- Database-backed result history.
- R2 raw trace browser.
- Blended leaderboard score.
- TTFT or ITL estimation from aggregate throughput.

These can be added later without changing the public route structure.
