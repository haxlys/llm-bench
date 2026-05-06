export type MetricStatus = "measured" | "directional" | "unavailable";

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString("en-US", {
    maximumFractionDigits: 2,
    minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
  });
}

export function statusLabel(status: MetricStatus): string {
  if (status === "unavailable") {
    return "not measured";
  }
  return status;
}

export function formatMetricValue(
  value: number | null,
  kind: "percent" | "number" | "tokensPerSecond" | "memoryGb",
): string {
  if (value === null) {
    return "not measured";
  }
  if (kind === "percent") {
    return formatPercent(value);
  }
  if (kind === "tokensPerSecond") {
    return `${formatNumber(value)} tok/s`;
  }
  if (kind === "memoryGb") {
    return `${formatNumber(value)} GB`;
  }
  return formatNumber(value);
}
