import type { AccuracyRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { caveatText, defaultLocale, messages, type Locale } from "../lib/i18n";
import { Badge } from "./Badge";

type AccuracyTableProps = {
  locale?: Locale;
  rows: AccuracyRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: AccuracyRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

export function AccuracyTable({ locale = defaultLocale, rows, variants }: AccuracyTableProps) {
  const t = messages[locale];

  return (
    <div className="table-scroll" role="region" aria-label={t.tables.accuracy.aria} tabIndex={0}>
      <table className="data-table accuracy-table">
        <colgroup>
          <col className="accuracy-col-model" />
          <col className="accuracy-col-task" />
          <col className="accuracy-col-metric" />
          <col className="accuracy-col-score" />
          <col className="accuracy-col-status" />
          <col className="accuracy-col-run" />
        </colgroup>
        <thead>
          <tr>
            <th>{t.tables.accuracy.headers.model}</th>
            <th>{t.tables.accuracy.headers.taskDim}</th>
            <th>{t.tables.accuracy.headers.metric}</th>
            <th>{t.tables.accuracy.headers.score}</th>
            <th>{t.tables.accuracy.headers.caveats}</th>
            <th>{t.tables.accuracy.headers.runId}</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="empty-cell">
                {t.tables.accuracy.empty}
              </td>
            </tr>
          ) : (
            rows.map((row) => {
              const variant = variants.get(row.variant);
              return (
                <tr key={`${row.variant}-${row.task}-${row.metric}-${row.runId}`}>
                  <td>
                    <strong>{variantLabel(row, variant)}</strong>
                    <div className="muted">{row.variant}</div>
                  </td>
                  <td>
                    <strong>{row.task}</strong>
                    <div className="muted">
                      {row.dim}
                      {row.subtask && row.subtask !== row.task ? ` / ${row.subtask}` : ""}
                    </div>
                  </td>
                  <td>{row.metric}</td>
                  <td className="numeric">{formatMetricValue(row.value, "percent", locale)}</td>
                  <td>
                    <Badge status={row.confidence} locale={locale} />
                    {row.caveats.length > 0 ? (
                      <div className="muted caveat-list">
                        {row.caveats.map((caveat) => caveatText(caveat, locale)).join(", ")}
                      </div>
                    ) : null}
                  </td>
                  <td className="run-id">{row.runId}</td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
