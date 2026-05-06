import { useMemo, useState } from "react";

import { AccuracyTable } from "../components/AccuracyTable";
import { benchmarkData, tasks, variantByKey } from "../lib/benchmark-data";
import { defaultLocale, messages, type Locale } from "../lib/i18n";

const allTasksOption = "all";
const allFamiliesOption = "all";

type AccuracyPageProps = {
  locale?: Locale;
};

export function AccuracyPage({ locale = defaultLocale }: AccuracyPageProps) {
  const t = messages[locale];
  const variants = useMemo(() => variantByKey(benchmarkData), []);
  const taskOptions = useMemo(() => tasks(benchmarkData), []);
  const familyOptions = useMemo(
    () =>
      Array.from(new Set(benchmarkData.variants.map((variant) => variant.family))).sort((a, b) =>
        a.localeCompare(b),
      ),
    [],
  );
  const [taskFilter, setTaskFilter] = useState(allTasksOption);
  const [familyFilter, setFamilyFilter] = useState(allFamiliesOption);

  const filteredRows = useMemo(
    () =>
      benchmarkData.accuracy
        .filter((row) => {
          if (taskFilter !== allTasksOption && row.task !== taskFilter) {
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
    [familyFilter, taskFilter, variants],
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
