import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { Badge } from "./Badge";

type SpeedScenarioTableProps = {
  rows: SpeedRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: SpeedRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

function formatPp(row: SpeedRow): string {
  if (row.backend === "mtplx" && row.ppTpsMean === 0) {
    return "not measured";
  }
  return formatMetricValue(row.ppTpsMean, "tokensPerSecond");
}

export function SpeedScenarioTable({ rows, variants }: SpeedScenarioTableProps) {
  return (
    <div className="table-scroll" role="region" aria-label="Speed scenario results" tabIndex={0}>
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
            <th>Model</th>
            <th>Scenario</th>
            <th>PP</th>
            <th>TG</th>
            <th>Memory</th>
            <th>Wall</th>
            <th>TTFT</th>
            <th>ITL</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={9} className="empty-cell">
                No speed rows match the current scenario.
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
                      {row.promptTokens} prompt / {row.generationTokens} gen
                    </div>
                  </td>
                  <td className="numeric">{formatPp(row)}</td>
                  <td className="numeric">{formatMetricValue(row.tgTpsMean, "tokensPerSecond")}</td>
                  <td className="numeric">{formatMetricValue(row.peakMemGbMean, "memoryGb")}</td>
                  <td className="numeric">{formatMetricValue(row.wallSecondsMean, "number")}s</td>
                  <td className="numeric">{formatMetricValue(row.ttftMs, "number")}</td>
                  <td className="numeric">{formatMetricValue(row.itlMs, "number")}</td>
                  <td>
                    <Badge status={row.confidence} />
                    {row.caveats.length > 0 ? (
                      <div className="muted caveat-list">{row.caveats.join(", ")}</div>
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
