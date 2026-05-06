import { defaultLocale, messages, type Locale } from "../lib/i18n";
import { Badge } from "./Badge";

type MetricGlossaryProps = {
  locale?: Locale;
};

export function MetricGlossary({ locale = defaultLocale }: MetricGlossaryProps) {
  const t = messages[locale].tables.glossary;

  return (
    <div className="table-scroll" role="region" aria-label={t.aria} tabIndex={0}>
      <table className="data-table glossary-table">
        <thead>
          <tr>
            <th>{t.headers.metric}</th>
            <th>{t.headers.status}</th>
            <th>{t.headers.interpretation}</th>
            <th>{t.headers.caveat}</th>
          </tr>
        </thead>
        <tbody>
          {t.items.map((item) => (
            <tr key={item.label}>
              <td>
                <strong>{item.label}</strong>
              </td>
              <td>
                <Badge status={item.status} locale={locale} />
              </td>
              <td>{item.summary}</td>
              <td className="muted">{item.caveat}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
