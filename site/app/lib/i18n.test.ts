import { describe, expect, it } from "vitest";

import {
  localeFromPathname,
  localizedPath,
  messages,
  switchLocalePath,
} from "./i18n";

describe("i18n path helpers", () => {
  it("detects default English and Korean paths without redirects", () => {
    expect(localeFromPathname("/")).toBe("en");
    expect(localeFromPathname("/speed")).toBe("en");
    expect(localeFromPathname("/ko")).toBe("ko");
    expect(localeFromPathname("/ko/speed")).toBe("ko");
  });

  it("builds localized page paths while keeping English as the default prefixless locale", () => {
    expect(localizedPath("/", "ko")).toBe("/ko");
    expect(localizedPath("/accuracy", "ko")).toBe("/ko/accuracy");
    expect(localizedPath("/ko/speed", "en")).toBe("/speed");
  });

  it("switches between matching English and Korean pages", () => {
    expect(switchLocalePath("/speed", "ko")).toBe("/ko/speed");
    expect(switchLocalePath("/ko/speed", "en")).toBe("/speed");
    expect(switchLocalePath("/ko", "en")).toBe("/");
  });

  it("does not localize static data asset paths", () => {
    expect(switchLocalePath("/data/benchmarks.json", "ko")).toBe("/data/benchmarks.json");
    expect(localizedPath("/data/summary.csv", "ko")).toBe("/data/summary.csv");
  });
});

describe("i18n messages", () => {
  it("keeps English and Korean message trees structurally aligned", () => {
    expect(shapeOf(messages.ko)).toEqual(shapeOf(messages.en));
  });
});

function shapeOf(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(shapeOf);
  }
  if (typeof value === "function") {
    return "function";
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, shapeOf(child)]),
    ) as Record<string, unknown>;
  }
  return typeof value;
}
