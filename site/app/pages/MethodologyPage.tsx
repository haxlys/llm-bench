import { MetricGlossary } from "../components/MetricGlossary";
import { benchmarkData, scenarios } from "../lib/benchmark-data";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

type MethodologyPageProps = {
  locale?: Locale;
};

export function MethodologyPage({ locale = defaultLocale }: MethodologyPageProps) {
  const t = messages[locale];
  const scenarioRows = scenarios(benchmarkData).map((scenario) => {
    const [promptPart, generationPart] = scenario.split("_");
    return {
      generationTokens: generationPart?.replace("g", "") ?? t.common.unknown,
      promptTokens: promptPart?.replace("p", "") ?? t.common.unknown,
      scenario,
    };
  });

  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="methodology-title">
        <div>
          <div className="eyebrow">{t.pages.methodology.eyebrow}</div>
          <h1 className="page-title" id="methodology-title">
            {t.pages.methodology.title}
          </h1>
          <p className="lead">{t.pages.methodology.lead}</p>
        </div>
        <dl className="report-meta panel" aria-label={t.pages.methodology.contextAria}>
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
          <div className="eyebrow">{t.pages.methodology.readingEyebrow}</div>
          <h2 id="reading-title">{t.pages.methodology.comparisonRulesTitle}</h2>
        </div>
        <ul className="dense-list">
          {t.pages.methodology.rules.map((rule) => (
            <li key={rule}>{rule}</li>
          ))}
        </ul>
      </section>

      <section className="section panel table-panel" aria-labelledby="glossary-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.methodology.metricsEyebrow}</div>
            <h2 id="glossary-title">{t.pages.methodology.metricGlossaryTitle}</h2>
          </div>
          <span className="muted">
            {benchmarkData.caveats.length} {t.common.caveatsTracked}
          </span>
        </div>
        <MetricGlossary locale={locale} />
      </section>

      <section className="section two-column">
        <div className="panel report-section" aria-labelledby="latency-title">
          <div>
            <div className="eyebrow">{t.pages.methodology.latencyEyebrow}</div>
            <h2 id="latency-title">{t.pages.methodology.latencyTitle}</h2>
          </div>
          <p>{t.pages.methodology.latencyBody(benchmarkData.benchVersion)}</p>
        </div>
        <div className="panel report-section" aria-labelledby="accuracy-caveat-title">
          <div>
            <div className="eyebrow">{t.pages.methodology.accuracyEyebrow}</div>
            <h2 id="accuracy-caveat-title">{t.pages.methodology.accuracyTitle}</h2>
          </div>
          <p>{t.pages.methodology.accuracyBody}</p>
        </div>
      </section>

      <section className="section panel table-panel" aria-labelledby="scenario-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.methodology.scenariosEyebrow}</div>
            <h2 id="scenario-title">{t.pages.methodology.scenarioTitle}</h2>
          </div>
          <span className="muted">
            {scenarioRows.length} {t.common.scenarios}
          </span>
        </div>
        <div className="table-scroll" role="region" aria-label={t.tables.scenario.aria} tabIndex={0}>
          <table className="data-table scenario-table">
            <thead>
              <tr>
                <th>{t.tables.scenario.headers.scenario}</th>
                <th>{t.tables.scenario.headers.promptTokens}</th>
                <th>{t.tables.scenario.headers.generationTokens}</th>
                <th>{t.tables.scenario.headers.use}</th>
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
                  <td className="muted">{t.tables.scenario.useText}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
