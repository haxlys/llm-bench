import type { ReactNode } from "react";

import { defaultLocale, type Locale } from "../lib/i18n";
import type { MetricStatus } from "../lib/format";
import { statusLabel } from "../lib/format";

type BadgeProps = {
  status?: MetricStatus;
  children?: ReactNode;
  locale?: Locale;
};

export function Badge({ status, children, locale = defaultLocale }: BadgeProps) {
  const className = status ? `badge ${status}` : "badge";
  return <span className={className}>{children ?? statusLabel(status ?? "measured", locale)}</span>;
}
