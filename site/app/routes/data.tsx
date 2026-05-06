import { createFileRoute } from "@tanstack/react-router";

import { DownloadLinks } from "../components/DownloadLinks";
import { benchmarkData } from "../lib/benchmark-data";

export const Route = createFileRoute("/data")({
  component: DataRoute,
});

function DataRoute() {
  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="data-title">
        <div>
          <div className="eyebrow">Data</div>
          <h1 className="page-title" id="data-title">
            Download benchmark artifacts
          </h1>
          <p className="lead">
            Public files mirror the committed CSV summaries and generated JSON export used by this
            site.
          </p>
        </div>
        <dl className="report-meta panel" aria-label="Data export context">
          <div>
            <dt>generatedAt</dt>
            <dd>{benchmarkData.generatedAt}</dd>
          </div>
          <div>
            <dt>sourceCommit</dt>
            <dd>{benchmarkData.sourceCommit || "unknown"}</dd>
          </div>
          <div>
            <dt>bench_version</dt>
            <dd>{benchmarkData.benchVersion}</dd>
          </div>
          <div>
            <dt>Rows</dt>
            <dd>
              {benchmarkData.accuracy.length} accuracy / {benchmarkData.speed.length} speed /{" "}
              {benchmarkData.mtplx.length} MTPLX
            </dd>
          </div>
        </dl>
      </section>

      <section className="section" aria-labelledby="downloads-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">Artifacts</div>
            <h2 id="downloads-title">Static downloads</h2>
          </div>
          <span className="muted">Served from /data</span>
        </div>
        <DownloadLinks />
      </section>
    </>
  );
}
