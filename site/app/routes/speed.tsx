import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { CaveatCallout } from "../components/CaveatCallout";
import { MtplxSpeedupPanel } from "../components/MtplxSpeedupPanel";
import { SpeedBars } from "../components/SpeedBars";
import { SpeedScenarioTable } from "../components/SpeedScenarioTable";
import { benchmarkData, scenarios, variantByKey } from "../lib/benchmark-data";

const preferredScenario = "p256_g128";

export const Route = createFileRoute("/speed")({
  component: SpeedRoute,
});

function SpeedRoute() {
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
          <div className="eyebrow">Speed explorer</div>
          <h1 className="page-title" id="speed-title">
            Token throughput, memory, and MTPLX speedups
          </h1>
          <p className="lead">
            Filter committed speed artifacts by prompt and generation shape. The report keeps PP,
            TG, memory, wall time, and MTPLX comparisons dense for benchmark review.
          </p>
        </div>
        <div className="row-count" aria-live="polite">
          <strong>{rows.length}</strong>
          <span className="muted"> of {benchmarkData.speed.length} rows</span>
        </div>
      </section>

      <section className="panel filter-panel" aria-label="Speed filters">
        <label className="filter-field" htmlFor="speed-scenario-filter">
          <span>Scenario</span>
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
              <div className="eyebrow">Top TG</div>
              <h2>Generation throughput</h2>
            </div>
            <span className="muted">Top 8 by tok/s</span>
          </div>
          <SpeedBars rows={rows.slice(0, 8)} variants={variants} />
        </div>
        <CaveatCallout title="TTFT and ITL">
          <p>
            TTFT and ITL are intentionally displayed from measured fields only. They remain{" "}
            <strong>not measured</strong> where bench_version {benchmarkData.benchVersion} exports
            null latency values.
          </p>
        </CaveatCallout>
      </section>

      <section className="section panel table-panel" aria-labelledby="speed-table-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">Explorer</div>
            <h2 id="speed-table-title">Speed results</h2>
          </div>
          <span className="muted">Sorted by TG tok/s</span>
        </div>
        <SpeedScenarioTable rows={rows} variants={variants} />
      </section>

      <section className="section" aria-labelledby="mtplx-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">MTPLX</div>
            <h2 id="mtplx-title">MTP-on versus AR baseline</h2>
          </div>
          <span className="muted">{benchmarkData.mtplx.length} comparisons</span>
        </div>
        <MtplxSpeedupPanel rows={benchmarkData.mtplx} variants={variants} />
      </section>
    </>
  );
}
