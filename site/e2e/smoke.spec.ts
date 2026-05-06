import { expect, test } from "@playwright/test";

const routes = [
  {
    path: "/",
    heading: /Local LLM results on/i,
    visibleText: "Accuracy snapshot",
  },
  {
    path: "/accuracy",
    heading: "Committed accuracy rows",
    visibleText: "Accuracy results",
  },
  {
    path: "/speed",
    heading: "Token throughput, memory, and MTPLX speedups",
    visibleText: "Speed results",
  },
  {
    path: "/methodology",
    heading: "How to read this benchmark report",
    visibleText: "Metric glossary",
  },
  {
    path: "/data",
    heading: "Download benchmark artifacts",
    visibleText: "Static downloads",
  },
] as const;

for (const route of routes) {
  test(`renders ${route.path}`, async ({ page }) => {
    await page.goto(route.path);

    await expect(page.getByRole("heading", { level: 1, name: route.heading })).toBeVisible();
    await expect(page.getByText(route.visibleText, { exact: true })).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();
  });
}

test("speed page marks TTFT and ITL as not measured", async ({ page }) => {
  await page.goto("/speed");

  const table = page.getByRole("region", { name: "Speed scenario results" });
  await expect(table.getByRole("columnheader", { name: "TTFT" })).toBeVisible();
  await expect(table.getByRole("columnheader", { name: "ITL" })).toBeVisible();
  await expect(table.getByText("not measured").first()).toBeVisible();
  await expect(
    page.getByText(/TTFT and ITL are intentionally displayed from measured fields only/i),
  ).toBeVisible();
});
