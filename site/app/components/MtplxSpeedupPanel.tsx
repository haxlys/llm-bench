import type { MtplxRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue, formatNumber } from "../lib/format";
import { Badge } from "./Badge";

type MtplxSpeedupPanelProps = {
  rows: MtplxRow[];
  variants: Map<string, Variant>;
};

function modelLabel(variantKey: string, variants: Map<string, Variant>): string {
  const variant = variants.get(variantKey);
  if (!variant) {
    return variantKey;
  }
  return `${variant.modelId} ${variant.quant}`;
}

export function MtplxSpeedupPanel({ rows, variants }: MtplxSpeedupPanelProps) {
  const sorted = rows.slice().sort((a, b) => b.speedup - a.speedup);

  return (
    <div className="mtplx-grid">
      {sorted.length === 0 ? (
        <div className="panel empty-cell">No MTPLX speedup rows are present.</div>
      ) : (
        sorted.map((row) => (
          <article className="panel mtplx-card" key={`${row.pairKey}-${row.scenario}`}>
            <div className="mtplx-card-header">
              <div>
                <div className="eyebrow">{row.scenario}</div>
                <h3>{formatNumber(row.speedup)}x speedup</h3>
              </div>
              <Badge status={row.confidence} />
            </div>
            <p className="muted mtplx-models">
              {modelLabel(row.mtpVariant, variants)} vs {modelLabel(row.arVariant, variants)}
            </p>
            <dl className="compact-dl">
              <div>
                <dt>MTP TG</dt>
                <dd>{formatMetricValue(row.mtpTgTpsMean, "tokensPerSecond")}</dd>
              </div>
              <div>
                <dt>AR TG</dt>
                <dd>{formatMetricValue(row.arTgTpsMean, "tokensPerSecond")}</dd>
              </div>
              <div>
                <dt>Accept d1</dt>
                <dd>{formatNumber(row.acceptance.d1)}</dd>
              </div>
              <div>
                <dt>Accept d2</dt>
                <dd>{formatNumber(row.acceptance.d2)}</dd>
              </div>
              <div>
                <dt>Accept d3</dt>
                <dd>{formatNumber(row.acceptance.d3)}</dd>
              </div>
              <div>
                <dt>Verify</dt>
                <dd>{formatMetricValue(row.verifyMsPerCallMean, "number")} ms</dd>
              </div>
            </dl>
            {row.caveats.length > 0 ? (
              <div className="muted caveat-list">{row.caveats.join(", ")}</div>
            ) : null}
          </article>
        ))
      )}
    </div>
  );
}
