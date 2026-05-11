import type { CoverageRow, Variant } from "../lib/benchmark-data";
import { defaultLocale, messages, type Locale } from "../lib/i18n";
import { Badge } from "./Badge";

type CoverageTableProps = {
  locale?: Locale;
  rows: CoverageRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: CoverageRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

export function CoverageTable({ locale = defaultLocale, rows, variants }: CoverageTableProps) {
  const t = messages[locale];

  return (
    <div className="table-scroll" role="region" aria-label={t.tables.coverage.aria} tabIndex={0}>
      <table className="data-table coverage-table">
        <colgroup>
          <col className="coverage-col-model" />
          <col className="coverage-col-task" />
          <col className="coverage-col-dim" />
          <col className="coverage-col-runner" />
          <col className="coverage-col-lane" />
          <col className="coverage-col-status" />
        </colgroup>
        <thead>
          <tr>
            <th>{t.tables.coverage.headers.model}</th>
            <th>{t.tables.coverage.headers.task}</th>
            <th>{t.tables.coverage.headers.dim}</th>
            <th>{t.tables.coverage.headers.runner}</th>
            <th>{t.tables.coverage.headers.lane}</th>
            <th>{t.tables.coverage.headers.status}</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="empty-cell">
                {t.tables.coverage.empty}
              </td>
            </tr>
          ) : (
            rows.map((row) => {
              const variant = variants.get(row.variant);
              return (
                <tr key={`${row.variant}-${row.task}-${row.runner}`}>
                  <td>
                    <strong>{variantLabel(row, variant)}</strong>
                    <div className="muted">{row.variant}</div>
                  </td>
                  <td>
                    <strong>{row.task}</strong>
                    <div className="muted">
                      {row.status === "speed_only"
                        ? row.lane
                        : row.supported
                          ? row.confidence
                          : "unsupported"}
                    </div>
                  </td>
                  <td>{row.dim}</td>
                  <td>{row.runner}</td>
                  <td>{row.lane}</td>
                  <td>
                    <Badge status={row.status} locale={locale} />
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
