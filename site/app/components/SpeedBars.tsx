import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";

type SpeedBarsProps = {
  rows: SpeedRow[];
  variants: Map<string, Variant>;
};

function variantLabel(row: SpeedRow, variant: Variant | undefined): string {
  if (!variant) {
    return row.modelId;
  }
  return `${variant.modelId} ${variant.quant}`;
}

export function SpeedBars({ rows, variants }: SpeedBarsProps) {
  const max = Math.max(...rows.map((row) => row.tgTpsMean), 1);

  return (
    <div className="speed-bars">
      {rows.map((row) => {
        const variant = variants.get(row.variant);
        return (
          <div className="speed-row" key={`${row.variant}-${row.scenario}`}>
            <div className="speed-row-label">
              <strong>{variantLabel(row, variant)}</strong>
              <span>{formatMetricValue(row.tgTpsMean, "tokensPerSecond")}</span>
            </div>
            <div className="bar-track" aria-hidden="true">
              <div className="bar-fill" style={{ width: `${(row.tgTpsMean / max) * 100}%` }} />
            </div>
            <div className="speed-row-meta">
              <span>{row.variant}</span>
              <span>{formatMetricValue(row.peakMemGbMean, "memoryGb")}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
