import type { SpeedRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue } from "../lib/format";
import { defaultLocale, type Locale } from "../lib/i18n";

type SpeedBarsProps = {
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

export function SpeedBars({ locale = defaultLocale, rows, variants }: SpeedBarsProps) {
  const max = Math.max(...rows.map((row) => row.tgTpsMean), 1);

  return (
    <div className="speed-bars">
      {rows.map((row) => {
        const variant = variants.get(row.variant);
        return (
          <div className="speed-row" key={`${row.variant}-${row.scenario}`}>
            <div className="speed-row-label">
              <strong>{variantLabel(row, variant)}</strong>
              <span>{formatMetricValue(row.tgTpsMean, "tokensPerSecond", locale)}</span>
            </div>
            <div className="bar-track" aria-hidden="true">
              <div className="bar-fill" style={{ width: `${(row.tgTpsMean / max) * 100}%` }} />
            </div>
            <div className="speed-row-meta">
              <span>{row.variant}</span>
              <span>{formatMetricValue(row.peakMemGbMean, "memoryGb", locale)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
