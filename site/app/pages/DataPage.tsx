import { DownloadLinks } from "../components/DownloadLinks";
import { benchmarkData } from "../lib/benchmark-data";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

type DataPageProps = {
  locale?: Locale;
};

export function DataPage({ locale = defaultLocale }: DataPageProps) {
  const t = messages[locale];

  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="data-title">
        <div>
          <div className="eyebrow">{t.pages.data.eyebrow}</div>
          <h1 className="page-title" id="data-title">
            {t.pages.data.title}
          </h1>
          <p className="lead">{t.pages.data.lead}</p>
        </div>
        <dl className="report-meta panel" aria-label={t.pages.data.contextAria}>
          <div>
            <dt>generatedAt</dt>
            <dd>{benchmarkData.generatedAt}</dd>
          </div>
          <div>
            <dt>sourceCommit</dt>
            <dd>{benchmarkData.sourceCommit || t.common.unknown}</dd>
          </div>
          <div>
            <dt>bench_version</dt>
            <dd>{benchmarkData.benchVersion}</dd>
          </div>
          <div>
            <dt>{t.pages.data.rowsLabel}</dt>
            <dd>
              {t.pages.data.rowsValue(
                benchmarkData.accuracy.length,
                benchmarkData.speed.length,
                benchmarkData.mtplx.length,
              )}
            </dd>
          </div>
        </dl>
      </section>

      <section className="section" aria-labelledby="downloads-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.data.artifactsEyebrow}</div>
            <h2 id="downloads-title">{t.pages.data.downloadsTitle}</h2>
          </div>
          <span className="muted">{t.pages.data.servedFrom}</span>
        </div>
        <DownloadLinks locale={locale} />
      </section>
    </>
  );
}
