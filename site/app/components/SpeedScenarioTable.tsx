import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { caveatText, defaultLocale, messages, type Locale } from "../lib/i18n";
import { Badge } from "./Badge";

type SpeedScenarioTableProps = {
  locale?: Locale;
  rows: SpeedRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: SpeedRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

function formatPp(row: SpeedRow, locale: Locale): string {
  if (row.backend === "mtplx" && row.ppTpsMean === 0) {
    return formatMetricValue(null, "tokensPerSecond", locale);
  }
  return formatMetricValue(row.ppTpsMean, "tokensPerSecond", locale);
}

export function SpeedScenarioTable({ locale = defaultLocale, rows, variants }: SpeedScenarioTableProps) {
  const t = messages[locale];

  return (
    <div className="table-scroll" role="region" aria-label={t.tables.speed.aria} tabIndex={0}>
      <table className="data-table speed-table">
        <colgroup>
          <col className="speed-col-model" />
          <col className="speed-col-scenario" />
          <col className="speed-col-metric" />
          <col className="speed-col-metric" />
          <col className="speed-col-metric" />
          <col className="speed-col-metric" />
          <col className="speed-col-metric" />
          <col className="speed-col-metric" />
          <col className="speed-col-status" />
        </colgroup>
        <thead>
          <tr>
            <th>{t.tables.speed.headers.model}</th>
            <th>{t.tables.speed.headers.scenario}</th>
            <th>{t.tables.speed.headers.pp}</th>
            <th>{t.tables.speed.headers.tg}</th>
            <th>{t.tables.speed.headers.memory}</th>
            <th>{t.tables.speed.headers.wall}</th>
            <th>{t.tables.speed.headers.ttft}</th>
            <th>{t.tables.speed.headers.itl}</th>
            <th>{t.tables.speed.headers.status}</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={9} className="empty-cell">
                {t.tables.speed.empty}
              </td>
            </tr>
          ) : (
            rows.map((row) => {
              const variant = variants.get(row.variant);
              return (
                <tr key={`${row.variant}-${row.scenario}`}>
                  <td>
                    <strong>{variantLabel(row, variant)}</strong>
                    <div className="muted">{row.variant}</div>
                  </td>
                  <td>
                    <strong>{row.scenario}</strong>
                    <div className="muted">
                      {row.promptTokens} {t.common.prompt} / {row.generationTokens} {t.common.gen}
                    </div>
                  </td>
                  <td className="numeric">{formatPp(row, locale)}</td>
                  <td className="numeric">
                    {formatMetricValue(row.tgTpsMean, "tokensPerSecond", locale)}
                  </td>
                  <td className="numeric">
                    {formatMetricValue(row.peakMemGbMean, "memoryGb", locale)}
                  </td>
                  <td className="numeric">
                    {formatMetricValue(row.wallSecondsMean, "number", locale)}s
                  </td>
                  <td className="numeric">{formatMetricValue(row.ttftMs, "number", locale)}</td>
                  <td className="numeric">{formatMetricValue(row.itlMs, "number", locale)}</td>
                  <td>
                    <Badge status={row.confidence} locale={locale} />
                    {row.caveats.length > 0 ? (
                      <div className="muted caveat-list">
                        {row.caveats.map((caveat) => caveatText(caveat, locale)).join(", ")}
                      </div>
                    ) : null}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
