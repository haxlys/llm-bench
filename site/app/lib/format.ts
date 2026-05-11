import { defaultLocale, messages, type Locale } from "./i18n";

export type MetricStatus =
  | "measured"
  | "directional"
  | "unavailable"
  | "optional"
  | "speed_only"
  | "missing"
  | "unsupported";

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number, locale: Locale = defaultLocale): string {
  return value.toLocaleString(locale === "ko" ? "ko-KR" : "en-US", {
    maximumFractionDigits: 2,
    minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
  });
}

export function statusLabel(status: MetricStatus, locale: Locale = defaultLocale): string {
  return messages[locale].status[status];
}

export function formatMetricValue(
  value: number | null,
  kind: "percent" | "number" | "tokensPerSecond" | "memoryGb",
  locale: Locale = defaultLocale,
): string {
  if (value === null) {
    return messages[locale].status.unavailable;
  }
  if (kind === "percent") {
    return formatPercent(value);
  }
  if (kind === "tokensPerSecond") {
    return `${formatNumber(value, locale)} tok/s`;
  }
  if (kind === "memoryGb") {
    return `${formatNumber(value, locale)} GB`;
  }
  return formatNumber(value, locale);
}
