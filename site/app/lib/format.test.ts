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
    expect(statusLabel("diagnostic")).toBe("diagnostic");
    expect(statusLabel("unavailable")).toBe("not measured");
    expect(statusLabel("optional")).toBe("optional");
    expect(statusLabel("speed_only")).toBe("speed-only");
    expect(statusLabel("missing")).toBe("missing");
    expect(statusLabel("measured", "ko")).toBe("측정됨");
    expect(statusLabel("directional", "ko")).toBe("방향성");
    expect(statusLabel("diagnostic", "ko")).toBe("진단용");
    expect(statusLabel("unavailable", "ko")).toBe("미측정");
    expect(statusLabel("optional", "ko")).toBe("옵션");
    expect(statusLabel("speed_only", "ko")).toBe("속도 전용");
    expect(statusLabel("missing", "ko")).toBe("누락");
  });
});
