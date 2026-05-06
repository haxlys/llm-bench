import { describe, expect, it } from "vitest";

import { formatMetricValue, formatNumber, formatPercent, statusLabel } from "./format";

describe("format helpers", () => {
  it("keeps measured zero distinct from unavailable null", () => {
    expect(formatMetricValue(0, "percent")).toBe("0.0%");
    expect(formatMetricValue(null, "percent")).toBe("not measured");
    expect(formatMetricValue(null, "percent", "ko")).toBe("미측정");
  });

  it("formats percent and numeric values consistently", () => {
    expect(formatPercent(0.90853)).toBe("90.9%");
    expect(formatNumber(112.154)).toBe("112.15");
    expect(formatMetricValue(112.154, "tokensPerSecond", "ko")).toBe("112.15 tok/s");
  });

  it("labels confidence statuses", () => {
    expect(statusLabel("measured")).toBe("measured");
    expect(statusLabel("directional")).toBe("directional");
    expect(statusLabel("unavailable")).toBe("not measured");
    expect(statusLabel("measured", "ko")).toBe("측정됨");
    expect(statusLabel("directional", "ko")).toBe("방향성");
    expect(statusLabel("unavailable", "ko")).toBe("미측정");
  });
});
