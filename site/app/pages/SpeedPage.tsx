import { useMemo, useState } from "react";

import { CaveatCallout } from "../components/CaveatCallout";
import { MtplxSpeedupPanel } from "../components/MtplxSpeedupPanel";
import { SpeedBars } from "../components/SpeedBars";
import { SpeedScenarioTable } from "../components/SpeedScenarioTable";
import { benchmarkData, scenarios, variantByKey } from "../lib/benchmark-data";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

const preferredScenario = "p256_g128";

type SpeedPageProps = {
  locale?: Locale;
};

export function SpeedPage({ locale = defaultLocale }: SpeedPageProps) {
  const t = messages[locale];
  const variants = useMemo(() => variantByKey(benchmarkData), []);
  const scenarioOptions = useMemo(() => scenarios(benchmarkData), []);
  const defaultScenario = scenarioOptions.includes(preferredScenario)
    ? preferredScenario
    : (scenarioOptions[0] ?? "");
  const [scenario, setScenario] = useState(defaultScenario);
  const rows = useMemo(
    () =>
      benchmarkData.speed
        .filter((row) => row.scenario === scenario)
        .slice()
        .sort((a, b) => {
          const tgComparison = b.tgTpsMean - a.tgTpsMean;
          if (tgComparison !== 0) {
            return tgComparison;
          }
          return a.modelId.localeCompare(b.modelId);
        }),
    [scenario],
  );

  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="speed-title">
        <div>
          <div className="eyebrow">{t.pages.speed.eyebrow}</div>
          <h1 className="page-title" id="speed-title">
            {t.pages.speed.title}
          </h1>
          <p className="lead">{t.pages.speed.lead}</p>
        </div>
        <div className="row-count" aria-live="polite">
          <strong>{rows.length}</strong>
          <span className="muted">
            {" "}
            {t.common.of} {benchmarkData.speed.length} {t.common.rows}
          </span>
        </div>
      </section>

      <section className="panel filter-panel" aria-label={t.pages.speed.filtersAria}>
        <label className="filter-field" htmlFor="speed-scenario-filter">
          <span>{t.pages.speed.scenarioLabel}</span>
          <select
            id="speed-scenario-filter"
            value={scenario}
            onChange={(event) => setScenario(event.target.value)}
          >
            {scenarioOptions.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="section two-column speed-overview-grid">
        <div className="panel">
          <div className="section-header">
            <div>
              <div className="eyebrow">{t.pages.speed.topTgEyebrow}</div>
              <h2>{t.pages.speed.generationTitle}</h2>
            </div>
            <span className="muted">{t.pages.speed.topByTok}</span>
          </div>
          <SpeedBars rows={rows.slice(0, 8)} variants={variants} locale={locale} />
        </div>
        <CaveatCallout title={t.pages.speed.latencyTitle}>
          <p>
            {t.pages.speed.latencyBodyStart}{" "}
            <strong>{messages[locale].status.unavailable}</strong>{" "}
            {t.pages.speed.latencyBodyEnd(benchmarkData.benchVersion)}
          </p>
        </CaveatCallout>
      </section>

      <section className="section panel table-panel" aria-labelledby="speed-table-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.speed.tableEyebrow}</div>
            <h2 id="speed-table-title">{t.pages.speed.tableTitle}</h2>
          </div>
          <span className="muted">{t.pages.speed.sortedBy}</span>
        </div>
        <SpeedScenarioTable rows={rows} variants={variants} locale={locale} />
      </section>

      <section className="section" aria-labelledby="mtplx-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.speed.mtplxEyebrow}</div>
            <h2 id="mtplx-title">{t.pages.speed.mtplxTitle}</h2>
          </div>
          <span className="muted">
            {benchmarkData.mtplx.length} {t.common.comparisons}
          </span>
        </div>
        <MtplxSpeedupPanel rows={benchmarkData.mtplx} variants={variants} locale={locale} />
      </section>
    </>
  );
}
