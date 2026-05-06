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
    <table className="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Task</th>
          <th>Metric</th>
          <th>Score</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const variant = variants.get(row.variant);
          return (
            <tr key={`${row.variant}-${row.task}-${row.metric}-${row.runId}`}>
              <td>
                <strong>{variantLabel(row, variant)}</strong>
                <div className="muted">{row.variant}</div>
              </td>
              <td>
                <strong>{row.task}</strong>
                <div className="muted">{row.subtask}</div>
              </td>
              <td>{row.metric}</td>
              <td>{formatMetricValue(row.value, "percent")}</td>
              <td>
                <Badge status={row.confidence} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
