import { Download } from "lucide-react";

import { defaultLocale, messages, type Locale } from "../lib/i18n";

type DownloadLinksProps = {
  locale?: Locale;
};

export function DownloadLinks({ locale = defaultLocale }: DownloadLinksProps) {
  const artifacts = messages[locale].downloads;

  return (
    <div className="download-grid">
      {artifacts.map((artifact) => (
        <a className="download-link panel" href={artifact.href} key={artifact.href}>
          <Download size={18} aria-hidden="true" />
          <span>
            <strong>{artifact.label}</strong>
            <small>{artifact.description}</small>
          </span>
        </a>
      ))}
    </div>
  );
}
