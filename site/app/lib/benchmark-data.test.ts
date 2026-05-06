import { describe, expect, it } from "vitest";

import {
  type BenchmarkData,
  benchmarkData,
  bestAccuracyRows,
  fastestSpeedRows,
  scenarios,
  tasks,
  variantByKey,
} from "./benchmark-data";

const fixture: BenchmarkData = {
  accuracy: [
    {
      caveats: [],
      confidence: "measured",
      dim: "code",
      metric: "pass@1",
      modelId: "model-a",
      runId: "run-a",
      stderr: 0.01,
      subtask: "humaneval",
      task: "humaneval",
      timestamp: "2026-05-01T00:00:00Z",
      value: 0.9,
      variant: "a",
    },
    {
      caveats: [],
      confidence: "measured",
      dim: "code",
      metric: "pass@1",
      modelId: "model-b",
      runId: "run-b",
      stderr: 0.02,
      subtask: "humaneval",
      task: "humaneval",
      timestamp: "2026-05-01T00:00:00Z",
      value: 0.8,
      variant: "b",
    },
  ],
  benchVersion: "0.3",
  caveats: [],
  generatedAt: "2026-05-06T00:00:00Z",
  hardware: {
    machine: "test-machine",
    memoryGb: 64,
    os: "test-os",
  },
  mtplx: [],
  sourceCommit: "test",
  speed: [
    {
      backend: "mlx",
      benchVersion: "0.3",
      caveats: [],
      confidence: "measured",
      firstMeasuredAt: "2026-05-01T00:00:00Z",
      fmt: "mlx",
      generationTokens: 128,
      itlMs: null,
      lastMeasuredAt: "2026-05-01T00:00:00Z",
      modelId: "model-a",
      peakMemGbMean: 20,
      ppTpsMean: 1000,
      promptTokens: 256,
      quant: "8bit",
      runIndices: [1],
      runs: 1,
      scenario: "p256_g128",
      tgTpsMean: 20,
      tier: "8bit",
      ttftMs: null,
      variant: "a",
      wallSecondsMean: 6.4,
    },
    {
      backend: "gguf",
      benchVersion: "0.3",
      caveats: [],
      confidence: "measured",
      firstMeasuredAt: "2026-05-01T00:00:00Z",
      fmt: "gguf",
      generationTokens: 128,
      itlMs: null,
      lastMeasuredAt: "2026-05-01T00:00:00Z",
      modelId: "model-b",
      peakMemGbMean: 18,
      ppTpsMean: 900,
      promptTokens: 256,
      quant: "q4",
      runIndices: [1],
      runs: 1,
      scenario: "p256_g128",
      tgTpsMean: 30,
      tier: "4bit",
      ttftMs: null,
      variant: "b",
      wallSecondsMean: 4.2,
    },
  ],
  variants: [
    {
      approxSizeGb: 10,
      architecture: "dense",
      artifactType: "hf_repo",
      backend: "mlx",
      family: "test",
      fmt: "mlx",
      generationMode: "",
      key: "a",
      modelId: "model-a",
      notes: "",
      paramsActiveB: 10,
      paramsTotalB: 10,
      quant: "8bit",
      tier: "8bit",
    },
    {
      approxSizeGb: 8,
      architecture: "dense",
      artifactType: "gguf",
      backend: "gguf",
      family: "test",
      fmt: "gguf",
      generationMode: "",
      key: "b",
      modelId: "model-b",
      notes: "",
      paramsActiveB: 8,
      paramsTotalB: 8,
      quant: "q4",
      tier: "4bit",
    },
  ],
};

describe("benchmark data helpers", () => {
  it("indexes variants by key", () => {
    expect(variantByKey(fixture).get("a")?.modelId).toBe("model-a");
  });

  it("returns best accuracy rows for a task", () => {
    expect(bestAccuracyRows(fixture, "humaneval", 1)[0].variant).toBe("a");
  });

  it("returns fastest speed rows for a scenario", () => {
    expect(fastestSpeedRows(fixture, "p256_g128", 1)[0].variant).toBe("b");
  });

  it("loads the generated benchmark dataset with compatible helpers", () => {
    expect(benchmarkData.benchVersion).toBeTruthy();
    expect(benchmarkData.generatedAt).toBeTruthy();
    expect(benchmarkData.hardware.machine).toBeTruthy();
    expect(benchmarkData.variants.length).toBeGreaterThan(0);
    expect(benchmarkData.accuracy.length).toBeGreaterThan(0);
    expect(benchmarkData.speed.length).toBeGreaterThan(0);
    expect(benchmarkData.mtplx.length).toBeGreaterThan(0);
    expect(benchmarkData.caveats.length).toBeGreaterThan(0);

    const firstVariant = benchmarkData.variants[0];
    const firstTask = tasks(benchmarkData)[0];
    const firstScenario = scenarios(benchmarkData)[0];
    const firstCaveat = benchmarkData.caveats[0];

    expect(variantByKey(benchmarkData).get(firstVariant.key)).toEqual(firstVariant);
    expect(bestAccuracyRows(benchmarkData, firstTask, 1)[0].task).toBe(firstTask);
    expect(fastestSpeedRows(benchmarkData, firstScenario, 1)[0].scenario).toBe(firstScenario);
    expect(firstCaveat.id).toBeTruthy();
    expect(["directional", "measured", "unavailable"]).toContain(firstCaveat.status);
  });
});
