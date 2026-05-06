import type { AccuracyRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { Badge } from "./Badge";

type AccuracyTableProps = {
  rows: AccuracyRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: AccuracyRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

export function AccuracyTable({ rows, variants }: AccuracyTableProps) {
  return (
    <div className="table-scroll" role="region" aria-label="Accuracy results" tabIndex={0}>
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
            <th>Model</th>
            <th>Task / dim</th>
            <th>Metric</th>
            <th>Score</th>
            <th>Status / caveats</th>
            <th>Run ID</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="empty-cell">
                No accuracy rows match the current filters.
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
                  <td className="numeric">{formatMetricValue(row.value, "percent")}</td>
                  <td>
                    <Badge status={row.confidence} />
                    {row.caveats.length > 0 ? (
                      <div className="muted caveat-list">{row.caveats.join(", ")}</div>
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
