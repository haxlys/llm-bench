import { expect, test } from "@playwright/test";

const routes = [
  {
    path: "/",
    heading: /Local LLM results on/i,
    locale: "en",
    visibleText: "Accuracy snapshot",
    navName: "Primary navigation",
  },
  {
    path: "/accuracy",
    heading: "Committed accuracy rows",
    locale: "en",
    visibleText: "Accuracy results",
    navName: "Primary navigation",
  },
  {
    path: "/speed",
    heading: "Token throughput, memory, and MTPLX speedups",
    locale: "en",
    visibleText: "Speed results",
    navName: "Primary navigation",
  },
  {
    path: "/methodology",
    heading: "How to read this benchmark report",
    locale: "en",
    visibleText: "Metric glossary",
    navName: "Primary navigation",
  },
  {
    path: "/data",
    heading: "Download benchmark artifacts",
    locale: "en",
    visibleText: "Static downloads",
    navName: "Primary navigation",
  },
  {
    path: "/ko",
    heading: /로컬 LLM 결과:/i,
    locale: "ko",
    visibleText: "Accuracy snapshot",
    navName: "주요 내비게이션",
  },
  {
    path: "/ko/accuracy",
    heading: "커밋된 정확도 행",
    locale: "ko",
    visibleText: "정확도 결과",
    navName: "주요 내비게이션",
  },
  {
    path: "/ko/speed",
    heading: "토큰 처리량, 메모리, MTPLX speedup",
    locale: "ko",
    visibleText: "속도 결과",
    navName: "주요 내비게이션",
  },
  {
    path: "/ko/methodology",
    heading: "이 벤치마크 리포트 읽는 법",
    locale: "ko",
    visibleText: "Metric glossary",
    navName: "주요 내비게이션",
  },
  {
    path: "/ko/data",
    heading: "Benchmark artifact 다운로드",
    locale: "ko",
    visibleText: "정적 다운로드",
    navName: "주요 내비게이션",
  },
] as const;

for (const route of routes) {
  test(`renders ${route.path}`, async ({ page }) => {
    await page.goto(route.path);

    await expect(page.getByRole("heading", { level: 1, name: route.heading })).toBeVisible();
    await expect(page.getByText(route.visibleText, { exact: true })).toBeVisible();
    await expect(page.getByRole("navigation", { name: route.navName })).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("lang", route.locale);
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

test("language switcher keeps the current page route", async ({ page }) => {
  await page.goto("/speed");
  await page.getByRole("navigation", { name: "Language" }).getByRole("link", { name: "KO" }).click();
  await expect(page).toHaveURL(/\/ko\/speed$/);
  await expect(
    page.getByRole("heading", { level: 1, name: "토큰 처리량, 메모리, MTPLX speedup" }),
  ).toBeVisible();

  await page.getByRole("navigation", { name: "언어" }).getByRole("link", { name: "EN" }).click();
  await expect(page).toHaveURL(/\/speed$/);
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "Token throughput, memory, and MTPLX speedups",
    }),
  ).toBeVisible();
});

test("Korean speed page marks TTFT and ITL as 미측정", async ({ page }) => {
  await page.goto("/ko/speed");

  const table = page.getByRole("region", { name: "속도 scenario 결과" });
  await expect(table.getByRole("columnheader", { name: "TTFT" })).toBeVisible();
  await expect(table.getByRole("columnheader", { name: "ITL" })).toBeVisible();
  await expect(table.getByText("미측정").first()).toBeVisible();
});

for (const artifactPath of [
  "/data/benchmarks.json",
  "/data/summary.csv",
  "/data/eval_summary_primary.csv",
  "/data/mtplx_speedups.csv",
] as const) {
  test(`serves ${artifactPath}`, async ({ request }) => {
    const response = await request.get(artifactPath);
    expect(response.ok()).toBe(true);
  });
}
