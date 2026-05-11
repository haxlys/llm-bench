import { AccuracyTable } from "../components/AccuracyTable";
import { CaveatCallout } from "../components/CaveatCallout";
import { FindingCard } from "../components/FindingCard";
import { SpeedBars } from "../components/SpeedBars";
import {
  benchmarkData,
  bestAccuracyRows,
  coverageSummary,
  fastestSpeedRows,
  variantByKey,
} from "../lib/benchmark-data";
import { formatMetricValue, formatPercent } from "../lib/format";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

type SummaryPageProps = {
  locale?: Locale;
};

export function SummaryPage({ locale = defaultLocale }: SummaryPageProps) {
  const t = messages[locale];
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
  const coverage = coverageSummary(benchmarkData);

  return (
    <>
      <section className="summary-hero" aria-labelledby="summary-title">
        <div>
          <div className="eyebrow">{t.pages.summary.eyebrow}</div>
          <h1 className="page-title" id="summary-title">
            {t.pages.summary.titlePrefix} {benchmarkData.hardware.machine}
          </h1>
          <p className="lead">
            {t.pages.summary.leadStart}{" "}
            <strong>{benchmarkData.sourceCommit || t.common.unknown}</strong>.{" "}
            {t.pages.summary.leadMiddle} {t.pages.summary.leadEnd}
          </p>
        </div>
        <dl className="summary-meta panel" aria-label={t.pages.summary.metadataAria}>
          <div>
            <dt>{t.pages.summary.meta.variants}</dt>
            <dd>{benchmarkData.variants.length}</dd>
          </div>
          <div>
            <dt>{t.pages.summary.meta.accuracyRows}</dt>
            <dd>{benchmarkData.accuracy.length}</dd>
          </div>
          <div>
            <dt>{t.pages.summary.meta.speedRows}</dt>
            <dd>{measuredSpeedRows.length}</dd>
          </div>
          <div>
            <dt>{t.pages.summary.meta.hardware}</dt>
            <dd>
              {benchmarkData.hardware.memoryGb}GB, {benchmarkData.hardware.os}
            </dd>
          </div>
        </dl>
      </section>

      <CaveatCallout title={t.pages.summary.coverageTitle}>
        <p>{t.pages.summary.coverageBody(coverage.missing, coverage.optional, coverage.speed_only)}</p>
      </CaveatCallout>

      <section className="section card-grid" aria-label={t.pages.summary.findingsAria}>
        <FindingCard
          label={t.pages.summary.findings.topHumanEval.label}
          value={
            bestAccuracy
              ? formatMetricValue(bestAccuracy.value, "percent", locale)
              : t.common.notAvailable
          }
          detail={
            bestAccuracy
              ? `${bestAccuracyVariant?.modelId ?? bestAccuracy.modelId} ${
                  bestAccuracyVariant?.quant ?? ""
                }`.trim()
              : t.pages.summary.findings.topHumanEval.empty
          }
        />
        <FindingCard
          label={t.pages.summary.findings.fastest.label}
          value={
            fastestRow
              ? formatMetricValue(fastestRow.tgTpsMean, "tokensPerSecond", locale)
              : t.common.notAvailable
          }
          detail={
            fastestRow
              ? t.pages.summary.findings.fastest.detail(
                  fastestVariant?.modelId ?? fastestRow.modelId,
                  formatMetricValue(fastestRow.peakMemGbMean, "memoryGb", locale),
                )
              : t.pages.summary.findings.fastest.empty
          }
        />
        <FindingCard
          label={t.pages.summary.findings.mtplx.label}
          value={mtplxBest ? `${mtplxBest.speedup.toFixed(2)}x` : t.common.notAvailable}
          detail={
            mtplxBest
              ? t.pages.summary.findings.mtplx.detail(
                  mtplxVariant?.modelId ?? mtplxBest.mtpVariant,
                  mtplxBest.scenario,
                  formatPercent(mtplxBest.acceptance.d1),
                )
              : t.pages.summary.findings.mtplx.empty
          }
        />
        <FindingCard
          label={t.pages.summary.findings.caveatCoverage.label}
          value={`${benchmarkData.caveats.length} ${
            t.pages.summary.findings.caveatCoverage.valueSuffix
          }`}
          detail={t.pages.summary.findings.caveatCoverage.detail}
        />
      </section>

      <CaveatCallout title={t.pages.summary.caveatTitle}>
        <p>{t.pages.summary.caveatBody}</p>
      </CaveatCallout>

      <section className="section two-column summary-data-grid">
        <div className="panel table-panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">{t.pages.summary.accuracySnapshot.eyebrow}</div>
              <h2>{t.pages.summary.accuracySnapshot.title}</h2>
            </div>
            <span className="muted">{t.pages.summary.accuracySnapshot.aside}</span>
          </div>
          <AccuracyTable rows={topAccuracy} variants={variants} locale={locale} />
        </div>
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">{t.pages.summary.throughput.eyebrow}</div>
              <h2>{t.pages.summary.throughput.title}</h2>
            </div>
            <span className="muted">{t.pages.summary.throughput.aside}</span>
          </div>
          <SpeedBars rows={fastest} variants={variants} locale={locale} />
        </div>
      </section>
    </>
  );
}
