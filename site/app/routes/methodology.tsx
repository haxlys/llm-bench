import { createFileRoute } from "@tanstack/react-router";

import { MetricGlossary } from "../components/MetricGlossary";
import { benchmarkData, scenarios } from "../lib/benchmark-data";

export const Route = createFileRoute("/methodology")({
  component: MethodologyRoute,
});

function MethodologyRoute() {
  const scenarioRows = scenarios(benchmarkData).map((scenario) => {
    const [promptPart, generationPart] = scenario.split("_");
    return {
      generationTokens: generationPart?.replace("g", "") ?? "unknown",
      promptTokens: promptPart?.replace("p", "") ?? "unknown",
      scenario,
    };
  });

  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="methodology-title">
        <div>
          <div className="eyebrow">Methodology</div>
          <h1 className="page-title" id="methodology-title">
            How to read this benchmark report
          </h1>
          <p className="lead">
            Results are static snapshots from committed benchmark artifacts. Treat scores as local
            Apple Silicon measurements for the listed hardware, benchmark version, and source
            commit.
          </p>
        </div>
        <dl className="report-meta panel" aria-label="Benchmark context">
          <div>
            <dt>bench_version</dt>
            <dd>{benchmarkData.benchVersion}</dd>
          </div>
          <div>
            <dt>Hardware</dt>
            <dd>{benchmarkData.hardware.machine}</dd>
          </div>
          <div>
            <dt>Memory</dt>
            <dd>{benchmarkData.hardware.memoryGb}GB</dd>
          </div>
          <div>
            <dt>OS</dt>
            <dd>{benchmarkData.hardware.os}</dd>
          </div>
        </dl>
      </section>

      <section className="section panel report-section" aria-labelledby="reading-title">
        <div>
          <div className="eyebrow">Reading order</div>
          <h2 id="reading-title">Comparison rules</h2>
        </div>
        <ul className="dense-list">
          <li>Compare speed rows within the same scenario only.</li>
          <li>Use TG tok/s as the primary throughput metric for generation speed.</li>
          <li>Use peak memory and wall time to identify tradeoffs hidden by throughput alone.</li>
          <li>Read accuracy by task; benchmarks with different prompts or graders are not pooled.</li>
          <li>Prefer measured rows over directional rows when drawing strict conclusions.</li>
        </ul>
      </section>

      <section className="section panel table-panel" aria-labelledby="glossary-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">Metrics</div>
            <h2 id="glossary-title">Metric glossary</h2>
          </div>
          <span className="muted">{benchmarkData.caveats.length} caveats tracked</span>
        </div>
        <MetricGlossary />
      </section>

      <section className="section two-column">
        <div className="panel report-section" aria-labelledby="latency-title">
          <div>
            <div className="eyebrow">Latency caveat</div>
            <h2 id="latency-title">TTFT and ITL unavailable</h2>
          </div>
          <p>
            TTFT and ITL are included as columns because they are useful latency metrics, but the
            current export records them as null. For bench_version {benchmarkData.benchVersion},
            latency comparisons should use TG tok/s, wall time, and memory instead.
          </p>
        </div>
        <div className="panel report-section" aria-labelledby="accuracy-caveat-title">
          <div>
            <div className="eyebrow">Accuracy caveat</div>
            <h2 id="accuracy-caveat-title">Directional exact-match rows</h2>
          </div>
          <p>
            Some generative tasks use exact-match or extraction-based scoring. Formatting,
            alternate equivalent answers, and answer extraction can undercount correct responses, so
            rows marked directional should be read as comparative signals, not final capability
            claims.
          </p>
        </div>
      </section>

      <section className="section panel table-panel" aria-labelledby="scenario-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">Speed scenarios</div>
            <h2 id="scenario-title">Prompt and generation matrix</h2>
          </div>
          <span className="muted">{scenarioRows.length} scenarios</span>
        </div>
        <div className="table-scroll" role="region" aria-label="Speed scenario matrix" tabIndex={0}>
          <table className="data-table scenario-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Prompt tokens</th>
                <th>Generation tokens</th>
                <th>Use</th>
              </tr>
            </thead>
            <tbody>
              {scenarioRows.map((row) => (
                <tr key={row.scenario}>
                  <td>
                    <strong>{row.scenario}</strong>
                  </td>
                  <td className="numeric">{row.promptTokens}</td>
                  <td className="numeric">{row.generationTokens}</td>
                  <td className="muted">Same-scenario speed comparison.</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
