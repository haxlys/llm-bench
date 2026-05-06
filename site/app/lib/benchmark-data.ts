import data from "../../src/data/benchmarks.json";

import type { MetricStatus } from "./format";

export type Variant = {
  approxSizeGb: number;
  architecture: string;
  artifactType: string;
  backend: string;
  family: string;
  fmt: string;
  generationMode: string;
  key: string;
  modelId: string;
  notes: string;
  paramsActiveB: number;
  paramsTotalB: number;
  quant: string;
  tier: string;
};

export type AccuracyRow = {
  caveats: string[];
  confidence: MetricStatus;
  dim: string;
  metric: string;
  modelId: string;
  runId: string;
  stderr: number;
  subtask: string;
  task: string;
  timestamp: string;
  value: number;
  variant: string;
};

export type SpeedRow = {
  backend: string;
  benchVersion: string;
  caveats: string[];
  confidence: MetricStatus;
  firstMeasuredAt: string;
  fmt: string;
  generationTokens: number;
  itlMs: number | null;
  lastMeasuredAt: string;
  modelId: string;
  peakMemGbMean: number;
  ppTpsMean: number;
  promptTokens: number;
  quant: string;
  runIndices: number[];
  runs: number;
  scenario: string;
  tgTpsMean: number;
  tier: string;
  ttftMs: number | null;
  variant: string;
  wallSecondsMean: number;
};

export type MtplxRow = {
  acceptance: {
    d1: number;
    d2: number;
    d3: number;
  };
  arPeakMemGbMean: number;
  arRuns: number;
  arTgTpsMean: number;
  arVariant: string;
  caveats: string[];
  confidence: MetricStatus;
  mtpPeakMemGbMean: number;
  mtpRuns: number;
  mtpTgTpsMean: number;
  mtpVariant: string;
  pairKey: string;
  scenario: string;
  speedup: number;
  verifyMsPerCallMean: number;
};

export type BenchmarkData = {
  accuracy: AccuracyRow[];
  benchVersion: string;
  caveats: Array<{
    id: string;
    status: MetricStatus;
  }>;
  generatedAt: string;
  hardware: {
    machine: string;
    memoryGb: number;
    os: string;
  };
  mtplx: MtplxRow[];
  sourceCommit: string;
  speed: SpeedRow[];
  variants: Variant[];
};

export const benchmarkData = data as BenchmarkData;

export function variantByKey(input: BenchmarkData): Map<string, Variant> {
  return new Map(input.variants.map((variant) => [variant.key, variant]));
}

export function bestAccuracyRows(
  input: BenchmarkData,
  task: string,
  limit: number,
): AccuracyRow[] {
  return input.accuracy
    .filter((row) => row.task === task)
    .slice()
    .sort((a, b) => b.value - a.value)
    .slice(0, limit);
}

export function fastestSpeedRows(
  input: BenchmarkData,
  scenario: string,
  limit: number,
): SpeedRow[] {
  return input.speed
    .filter((row) => row.scenario === scenario)
    .slice()
    .sort((a, b) => b.tgTpsMean - a.tgTpsMean)
    .slice(0, limit);
}

export function scenarios(input: BenchmarkData): string[] {
  return Array.from(new Set(input.speed.map((row) => row.scenario))).sort();
}

export function tasks(input: BenchmarkData): string[] {
  return Array.from(new Set(input.accuracy.map((row) => row.task))).sort();
}
