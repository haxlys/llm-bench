import { useMemo, useState } from "react";

import { AccuracyTable } from "../components/AccuracyTable";
import { CoverageTable } from "../components/CoverageTable";
import { benchmarkData, dimensions, tasks, variantByKey } from "../lib/benchmark-data";
import type { MetricStatus } from "../lib/format";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

const allTasksOption = "all";
const allFamiliesOption = "all";
const allDimensionsOption = "all";
const coverageStatuses: MetricStatus[] = [
  "measured",
  "directional",
  "diagnostic",
  "missing",
  "optional",
  "speed_only",
  "unsupported",
];

type AccuracyPageProps = {
  locale?: Locale;
};

export function AccuracyPage({ locale = defaultLocale }: AccuracyPageProps) {
  const t = messages[locale];
  const variants = useMemo(() => variantByKey(benchmarkData), []);
  const taskOptions = useMemo(() => tasks(benchmarkData), []);
  const dimensionOptions = useMemo(() => dimensions(benchmarkData), []);
  const familyOptions = useMemo(
    () =>
      Array.from(new Set(benchmarkData.variants.map((variant) => variant.family))).sort((a, b) =>
        a.localeCompare(b),
      ),
    [],
  );
  const [taskFilter, setTaskFilter] = useState(allTasksOption);
  const [dimensionFilter, setDimensionFilter] = useState(allDimensionsOption);
  const [familyFilter, setFamilyFilter] = useState(allFamiliesOption);

  const coverageRows = useMemo(
    () =>
      benchmarkData.coverage
        .filter((row) => {
          if (taskFilter !== allTasksOption && row.task !== taskFilter) {
            return false;
          }
          if (dimensionFilter !== allDimensionsOption && row.dim !== dimensionFilter) {
            return false;
          }
          return familyFilter === allFamiliesOption || row.family === familyFilter;
        })
        .slice()
        .sort((a, b) => {
          const statusComparison = coverageStatuses.indexOf(a.status) - coverageStatuses.indexOf(b.status);
          if (statusComparison !== 0) {
            return statusComparison;
          }
          const dimComparison = a.dim.localeCompare(b.dim);
          if (dimComparison !== 0) {
            return dimComparison;
          }
          const taskComparison = a.task.localeCompare(b.task);
          if (taskComparison !== 0) {
            return taskComparison;
          }
          return a.modelId.localeCompare(b.modelId);
        }),
    [dimensionFilter, familyFilter, taskFilter],
  );

  const coverageCounts = useMemo(
    () =>
      coverageStatuses.map((status) => ({
        status,
        value: coverageRows.filter((row) => row.status === status).length,
      })),
    [coverageRows],
  );

  const filteredRows = useMemo(
    () =>
      benchmarkData.accuracy
        .filter((row) => {
          if (taskFilter !== allTasksOption && row.task !== taskFilter) {
            return false;
          }
          if (dimensionFilter !== allDimensionsOption && row.dim !== dimensionFilter) {
            return false;
          }
          const variant = variants.get(row.variant);
          return familyFilter === allFamiliesOption || variant?.family === familyFilter;
        })
        .slice()
        .sort((a, b) => {
          const taskComparison = a.task.localeCompare(b.task);
          if (taskComparison !== 0) {
            return taskComparison;
          }
          const scoreComparison = b.value - a.value;
          if (scoreComparison !== 0) {
            return scoreComparison;
          }
          return a.modelId.localeCompare(b.modelId);
        }),
    [dimensionFilter, familyFilter, taskFilter, variants],
  );

  return (
    <>
      <section className="section-header explorer-header" aria-labelledby="accuracy-title">
        <div>
          <div className="eyebrow">{t.pages.accuracy.eyebrow}</div>
          <h1 className="page-title" id="accuracy-title">
            {t.pages.accuracy.title}
          </h1>
          <p className="lead">{t.pages.accuracy.lead}</p>
        </div>
        <div className="row-count" aria-live="polite">
          <strong>{filteredRows.length}</strong>
          <span className="muted">
            {" "}
            {t.common.of} {benchmarkData.accuracy.length} {t.common.rows}
          </span>
        </div>
      </section>

      <section className="panel filter-panel" aria-label={t.pages.accuracy.filtersAria}>
        <label className="filter-field" htmlFor="accuracy-task-filter">
          <span>{t.pages.accuracy.taskLabel}</span>
          <select
            id="accuracy-task-filter"
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value)}
          >
            <option value={allTasksOption}>{t.common.allTasks}</option>
            {taskOptions.map((task) => (
              <option key={task} value={task}>
                {task}
              </option>
            ))}
          </select>
        </label>
        <label className="filter-field" htmlFor="accuracy-family-filter">
          <span>{t.pages.accuracy.familyLabel}</span>
          <select
            id="accuracy-family-filter"
            value={familyFilter}
            onChange={(event) => setFamilyFilter(event.target.value)}
          >
            <option value={allFamiliesOption}>{t.common.allFamilies}</option>
            {familyOptions.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>
        <label className="filter-field" htmlFor="accuracy-dimension-filter">
          <span>{t.pages.accuracy.dimensionLabel}</span>
          <select
            id="accuracy-dimension-filter"
            value={dimensionFilter}
            onChange={(event) => setDimensionFilter(event.target.value)}
          >
            <option value={allDimensionsOption}>{t.common.allDimensions}</option>
            {dimensionOptions.map((dimension) => (
              <option key={dimension} value={dimension}>
                {dimension}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="section panel table-panel" aria-labelledby="coverage-table-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.accuracy.coverageEyebrow}</div>
            <h2 id="coverage-table-title">{t.pages.accuracy.coverageTitle}</h2>
          </div>
          <span className="muted">{t.pages.accuracy.coverageAside}</span>
        </div>
        <div className="coverage-summary-grid" aria-label={t.pages.accuracy.coverageTitle}>
          {coverageCounts.map((item) => (
            <div className="coverage-summary-item" key={item.status}>
              <span className={`badge ${item.status}`}>{t.status[item.status]}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
        <CoverageTable rows={coverageRows} variants={variants} locale={locale} />
      </section>

      <section className="section panel table-panel" aria-labelledby="accuracy-table-title">
        <div className="section-header">
          <div>
            <div className="eyebrow">{t.pages.accuracy.tableEyebrow}</div>
            <h2 id="accuracy-table-title">{t.pages.accuracy.tableTitle}</h2>
          </div>
          <span className="muted">{t.pages.accuracy.sortedBy}</span>
        </div>
        <AccuracyTable rows={filteredRows} variants={variants} locale={locale} />
      </section>
    </>
  );
}
