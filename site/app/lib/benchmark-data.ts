import data from "../../src/data/benchmarks.json";

import type { MetricStatus } from "./format";

const metricStatuses: ReadonlySet<string> = new Set<MetricStatus>([
  "measured",
  "directional",
  "unavailable",
]);

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
  paramsActiveB: number | null;
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
  stderr: number | null;
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

export const benchmarkData = validateBenchmarkData(data);

function validateBenchmarkData(input: unknown): BenchmarkData {
  assertBenchmarkData(input);
  return input;
}

function assertBenchmarkData(input: unknown): asserts input is BenchmarkData {
  assertRecord(input, "benchmark data");
  assertArray(input.accuracy, "accuracy");
  assertArray(input.speed, "speed");
  assertArray(input.mtplx, "mtplx");
  assertArray(input.variants, "variants");
  assertArray(input.caveats, "caveats");
  assertString(input.benchVersion, "benchVersion");
  assertString(input.generatedAt, "generatedAt");
  assertString(input.sourceCommit, "sourceCommit");
  assertRecord(input.hardware, "hardware");
  assertString(input.hardware.machine, "hardware.machine");
  assertNumber(input.hardware.memoryGb, "hardware.memoryGb");
  assertString(input.hardware.os, "hardware.os");
  input.accuracy.forEach(assertAccuracyRow);
  input.speed.forEach(assertSpeedRow);
  input.mtplx.forEach(assertMtplxRow);
  input.variants.forEach(assertVariant);
  input.caveats.forEach(assertCaveat);
}

function assertCaveat(input: unknown): asserts input is BenchmarkData["caveats"][number] {
  assertRecord(input, "caveat");
  assertString(input.id, "caveat id");
  assertMetricStatus(input.status, "caveat status");
}

function assertAccuracyRow(input: unknown): asserts input is AccuracyRow {
  assertRecord(input, "accuracy row");
  assertArray(input.caveats, "accuracy row caveats");
  input.caveats.forEach((caveat) => assertString(caveat, "accuracy row caveat"));
  assertMetricStatus(input.confidence, "accuracy row confidence");
  assertString(input.dim, "accuracy row dim");
  assertString(input.metric, "accuracy row metric");
  assertString(input.modelId, "accuracy row modelId");
  assertString(input.runId, "accuracy row runId");
  assertNullableNumber(input.stderr, "accuracy row stderr");
  assertString(input.subtask, "accuracy row subtask");
  assertString(input.task, "accuracy row task");
  assertString(input.timestamp, "accuracy row timestamp");
  assertNumber(input.value, "accuracy row value");
  assertString(input.variant, "accuracy row variant");
}

function assertSpeedRow(input: unknown): asserts input is SpeedRow {
  assertRecord(input, "speed row");
  assertString(input.backend, "speed row backend");
  assertString(input.benchVersion, "speed row benchVersion");
  assertArray(input.caveats, "speed row caveats");
  input.caveats.forEach((caveat) => assertString(caveat, "speed row caveat"));
  assertMetricStatus(input.confidence, "speed row confidence");
  assertString(input.firstMeasuredAt, "speed row firstMeasuredAt");
  assertString(input.fmt, "speed row fmt");
  assertNumber(input.generationTokens, "speed row generationTokens");
  assertNullableNumber(input.itlMs, "speed row itlMs");
  assertString(input.lastMeasuredAt, "speed row lastMeasuredAt");
  assertString(input.modelId, "speed row modelId");
  assertNumber(input.peakMemGbMean, "speed row peakMemGbMean");
  assertNumber(input.ppTpsMean, "speed row ppTpsMean");
  assertNumber(input.promptTokens, "speed row promptTokens");
  assertString(input.quant, "speed row quant");
  assertArray(input.runIndices, "speed row runIndices");
  input.runIndices.forEach((runIndex) => assertNumber(runIndex, "speed row runIndex"));
  assertNumber(input.runs, "speed row runs");
  assertString(input.scenario, "speed row scenario");
  assertNumber(input.tgTpsMean, "speed row tgTpsMean");
  assertString(input.tier, "speed row tier");
  assertNullableNumber(input.ttftMs, "speed row ttftMs");
  assertString(input.variant, "speed row variant");
  assertNumber(input.wallSecondsMean, "speed row wallSecondsMean");
}

function assertMtplxRow(input: unknown): asserts input is MtplxRow {
  assertRecord(input, "mtplx row");
  assertRecord(input.acceptance, "mtplx row acceptance");
  assertNumber(input.acceptance.d1, "mtplx row acceptance d1");
  assertNumber(input.acceptance.d2, "mtplx row acceptance d2");
  assertNumber(input.acceptance.d3, "mtplx row acceptance d3");
  assertNumber(input.arPeakMemGbMean, "mtplx row arPeakMemGbMean");
  assertNumber(input.arRuns, "mtplx row arRuns");
  assertNumber(input.arTgTpsMean, "mtplx row arTgTpsMean");
  assertString(input.arVariant, "mtplx row arVariant");
  assertArray(input.caveats, "mtplx row caveats");
  input.caveats.forEach((caveat) => assertString(caveat, "mtplx row caveat"));
  assertMetricStatus(input.confidence, "mtplx row confidence");
  assertNumber(input.mtpPeakMemGbMean, "mtplx row mtpPeakMemGbMean");
  assertNumber(input.mtpRuns, "mtplx row mtpRuns");
  assertNumber(input.mtpTgTpsMean, "mtplx row mtpTgTpsMean");
  assertString(input.mtpVariant, "mtplx row mtpVariant");
  assertString(input.pairKey, "mtplx row pairKey");
  assertString(input.scenario, "mtplx row scenario");
  assertNumber(input.speedup, "mtplx row speedup");
  assertNumber(input.verifyMsPerCallMean, "mtplx row verifyMsPerCallMean");
}

function assertVariant(input: unknown): asserts input is Variant {
  assertRecord(input, "variant");
  assertNumber(input.approxSizeGb, "variant approxSizeGb");
  assertString(input.architecture, "variant architecture");
  assertString(input.artifactType, "variant artifactType");
  assertString(input.backend, "variant backend");
  assertString(input.family, "variant family");
  assertString(input.fmt, "variant fmt");
  assertString(input.generationMode, "variant generationMode");
  assertString(input.key, "variant key");
  assertString(input.modelId, "variant modelId");
  assertString(input.notes, "variant notes");
  assertNullableNumber(input.paramsActiveB, "variant paramsActiveB");
  assertNumber(input.paramsTotalB, "variant paramsTotalB");
  assertString(input.quant, "variant quant");
  assertString(input.tier, "variant tier");
}

function assertRecord(input: unknown, label: string): asserts input is Record<string, unknown> {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new Error(`Expected ${label} to be an object`);
  }
}

function assertArray(input: unknown, label: string): asserts input is unknown[] {
  if (!Array.isArray(input)) {
    throw new Error(`Expected ${label} to be an array`);
  }
}

function assertString(input: unknown, label: string): asserts input is string {
  if (typeof input !== "string") {
    throw new Error(`Expected ${label} to be a string`);
  }
}

function assertNumber(input: unknown, label: string): asserts input is number {
  if (typeof input !== "number") {
    throw new Error(`Expected ${label} to be a number`);
  }
}

function assertNullableNumber(input: unknown, label: string): asserts input is number | null {
  if (input !== null && typeof input !== "number") {
    throw new Error(`Expected ${label} to be a number or null`);
  }
}

function assertMetricStatus(input: unknown, label: string): asserts input is MetricStatus {
  if (typeof input !== "string" || !metricStatuses.has(input)) {
    throw new Error(`Expected ${label} to be a metric status`);
  }
}

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
