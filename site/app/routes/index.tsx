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
import { formatMetricValue, formatPercent } from "../lib/format";

export const Route = createFileRoute("/")({
  component: SummaryRoute,
});

function SummaryRoute() {
  const variants = variantByKey(benchmarkData);
  const topAccuracy = bestAccuracyRows(benchmarkData, "humaneval", 5);
  const fastest = fastestSpeedRows(benchmarkData, "p256_g128", 5);
  const mtplxBest = benchmarkData.mtplx.slice().sort((a, b) => b.speedup - a.speedup)[0];
  const bestAccuracy = topAccuracy[0];
  const fastestRow = fastest[0];
  const bestAccuracyVariant = bestAccuracy ? variants.get(bestAccuracy.variant) : undefined;
  const fastestVariant = fastestRow ? variants.get(fastestRow.variant) : undefined;
  const mtplxVariant = mtplxBest ? variants.get(mtplxBest.mtpVariant) : undefined;
  const measuredSpeedRows = benchmarkData.speed.filter((row) => row.confidence === "measured");

  return (
    <>
      <section className="summary-hero" aria-labelledby="summary-title">
        <div>
          <div className="eyebrow">Benchmark report</div>
          <h1 className="page-title" id="summary-title">
            Local LLM results on {benchmarkData.hardware.machine}
          </h1>
          <p className="lead">
            Static report generated from committed benchmark artifacts at source commit{" "}
            <strong>{benchmarkData.sourceCommit || "unknown"}</strong>. Current coverage includes
            accuracy, prompt/generation throughput, peak memory, wall time, and MTPLX speedups.
          </p>
        </div>
        <dl className="summary-meta panel" aria-label="Dataset metadata">
          <div>
            <dt>Variants</dt>
            <dd>{benchmarkData.variants.length}</dd>
          </div>
          <div>
            <dt>Accuracy rows</dt>
            <dd>{benchmarkData.accuracy.length}</dd>
          </div>
          <div>
            <dt>Speed rows</dt>
            <dd>{measuredSpeedRows.length}</dd>
          </div>
          <div>
            <dt>Hardware</dt>
            <dd>
              {benchmarkData.hardware.memoryGb}GB, {benchmarkData.hardware.os}
            </dd>
          </div>
        </dl>
      </section>

      <section className="section card-grid" aria-label="Key findings">
        <FindingCard
          label="Top HumanEval"
          value={
            bestAccuracy
              ? formatMetricValue(bestAccuracy.value, "percent")
              : "not available"
          }
          detail={
            bestAccuracy
              ? `${bestAccuracyVariant?.modelId ?? bestAccuracy.modelId} ${bestAccuracyVariant?.quant ?? ""}`.trim()
              : "No HumanEval rows are present in the current data export."
          }
        />
        <FindingCard
          label="Fastest p256_g128"
          value={
            fastestRow
              ? formatMetricValue(fastestRow.tgTpsMean, "tokensPerSecond")
              : "not available"
          }
          detail={
            fastestRow
              ? `${fastestVariant?.modelId ?? fastestRow.modelId} at ${formatMetricValue(
                  fastestRow.peakMemGbMean,
                  "memoryGb",
                )} peak memory.`
              : "No p256_g128 speed rows are present in the current data export."
          }
        />
        <FindingCard
          label="MTPLX speedup"
          value={mtplxBest ? `${mtplxBest.speedup.toFixed(2)}x` : "not available"}
          detail={
            mtplxBest
              ? `${mtplxVariant?.modelId ?? mtplxBest.mtpVariant} on ${mtplxBest.scenario}, with d1 acceptance ${formatPercent(
                  mtplxBest.acceptance.d1,
                )}.`
              : "No MTPLX comparison rows are present in the current data export."
          }
        />
        <FindingCard
          label="Caveat coverage"
          value={`${benchmarkData.caveats.length} tracked`}
          detail="Latency metrics and generative exact-match interpretation are explicitly marked where applicable."
        />
      </section>

      <CaveatCallout title="Metric caveat">
        <p>
          TTFT and ITL are unavailable in this export, so speed comparisons use generation tokens
          per second. Generative exact-match accuracy rows can be directional when answer extraction
          or formatting may undercount correct outputs.
        </p>
      </CaveatCallout>

      <section className="section two-column summary-data-grid">
        <div className="panel table-panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">Accuracy snapshot</div>
              <h2>Top HumanEval rows</h2>
            </div>
            <span className="muted">Top 5 by score</span>
          </div>
          <AccuracyTable rows={topAccuracy} variants={variants} />
        </div>
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">Generation throughput</div>
              <h2>Fastest p256_g128 rows</h2>
            </div>
            <span className="muted">Top 5 by TG tok/s</span>
          </div>
          <SpeedBars rows={fastest} variants={variants} />
        </div>
      </section>
    </>
  );
}
