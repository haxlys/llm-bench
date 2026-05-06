import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { AccuracyTable } from "../components/AccuracyTable";
import { benchmarkData, tasks, variantByKey } from "../lib/benchmark-data";

const allTasksOption = "all";
const allFamiliesOption = "all";

export const Route = createFileRoute("/accuracy")({
  component: AccuracyRoute,
});

function AccuracyRoute() {
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
          <div className="eyebrow">Accuracy explorer</div>
          <h1 className="page-title" id="accuracy-title">
            Committed accuracy rows
          </h1>
          <p className="lead">
            Filter committed benchmark accuracy artifacts by task and model family.
          </p>
        </div>
        <div className="row-count" aria-live="polite">
          <strong>{filteredRows.length}</strong>
          <span className="muted"> of {benchmarkData.accuracy.length} rows</span>
        </div>
      </section>

      <section className="panel filter-panel" aria-label="Accuracy filters">
        <label className="filter-field" htmlFor="accuracy-task-filter">
          <span>Task</span>
          <select
            id="accuracy-task-filter"
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value)}
          >
            <option value={allTasksOption}>All tasks</option>
            {taskOptions.map((task) => (
              <option key={task} value={task}>
                {task}
              </option>
            ))}
          </select>
        </label>
        <label className="filter-field" htmlFor="accuracy-family-filter">
          <span>Family</span>
          <select
            id="accuracy-family-filter"
            value={familyFilter}
            onChange={(event) => setFamilyFilter(event.target.value)}
          >
            <option value={allFamiliesOption}>All families</option>
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
            <div className="eyebrow">Explorer</div>
            <h2 id="accuracy-table-title">Accuracy results</h2>
          </div>
          <span className="muted">Sorted by task, score, model</span>
        </div>
        <AccuracyTable rows={filteredRows} variants={variants} />
      </section>
    </>
  );
}
