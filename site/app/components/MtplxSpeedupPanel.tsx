import type { MtplxRow, Variant } from "../lib/benchmark-data";
import { formatMetricValue, formatNumber, formatPercent } from "../lib/format";
import { caveatText, defaultLocale, messages, type Locale } from "../lib/i18n";
import { Badge } from "./Badge";

type MtplxSpeedupPanelProps = {
  locale?: Locale;
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

export function MtplxSpeedupPanel({
  locale = defaultLocale,
  rows,
  variants,
}: MtplxSpeedupPanelProps) {
  const sorted = rows.slice().sort((a, b) => b.speedup - a.speedup);
  const t = messages[locale];

  return (
    <div className="mtplx-grid">
      {sorted.length === 0 ? (
        <div className="panel empty-cell">{t.mtplx.empty}</div>
      ) : (
        sorted.map((row) => (
          <article className="panel mtplx-card" key={`${row.pairKey}-${row.scenario}`}>
            <div className="mtplx-card-header">
              <div>
                <div className="eyebrow">{row.scenario}</div>
                <h3>
                  {formatNumber(row.speedup, locale)}x {t.mtplx.speedupSuffix}
                </h3>
              </div>
              <Badge status={row.confidence} locale={locale} />
            </div>
            <p className="muted mtplx-models">
              {modelLabel(row.mtpVariant, variants)} {t.mtplx.versus}{" "}
              {modelLabel(row.arVariant, variants)}
            </p>
            <dl className="compact-dl">
              <div>
                <dt>{t.mtplx.labels.mtpTg}</dt>
                <dd>{formatMetricValue(row.mtpTgTpsMean, "tokensPerSecond", locale)}</dd>
              </div>
              <div>
                <dt>{t.mtplx.labels.arTg}</dt>
                <dd>{formatMetricValue(row.arTgTpsMean, "tokensPerSecond", locale)}</dd>
              </div>
              <div>
                <dt>{t.mtplx.labels.acceptD1}</dt>
                <dd>{formatPercent(row.acceptance.d1)}</dd>
              </div>
              <div>
                <dt>{t.mtplx.labels.acceptD2}</dt>
                <dd>{formatPercent(row.acceptance.d2)}</dd>
              </div>
              <div>
                <dt>{t.mtplx.labels.acceptD3}</dt>
                <dd>{formatPercent(row.acceptance.d3)}</dd>
              </div>
              <div>
                <dt>{t.mtplx.labels.verify}</dt>
                <dd>{formatMetricValue(row.verifyMsPerCallMean, "number", locale)} ms</dd>
              </div>
            </dl>
            {row.caveats.length > 0 ? (
              <div className="muted caveat-list">
                {row.caveats.map((caveat) => caveatText(caveat, locale)).join(", ")}
              </div>
            ) : null}
          </article>
        ))
      )}
    </div>
  );
}
