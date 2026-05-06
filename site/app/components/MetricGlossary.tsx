type MetricGlossaryItem = {
  caveat: string;
  label: string;
  status: "measured" | "directional" | "unavailable";
  summary: string;
};

const metricGlossary: MetricGlossaryItem[] = [
  {
    caveat: "Measured from benchmark run summaries.",
    label: "PP tok/s",
    status: "measured",
    summary: "Prompt processing throughput. Higher means faster prompt ingestion.",
  },
  {
    caveat: "Primary speed comparison metric for this export.",
    label: "TG tok/s",
    status: "measured",
    summary: "Generation throughput during decode. Higher means faster output tokens.",
  },
  {
    caveat: "Reported as peak resident memory across runs.",
    label: "Peak memory",
    status: "measured",
    summary: "Mean peak memory in GB for the benchmark scenario.",
  },
  {
    caveat: "End-to-end scenario duration.",
    label: "Wall time",
    status: "measured",
    summary: "Mean elapsed seconds for the prompt and generation scenario.",
  },
  {
    caveat: "bench_version 0.3 exports null latency fields.",
    label: "TTFT",
    status: "unavailable",
    summary: "Time to first token in milliseconds. Not measured in the current export.",
  },
  {
    caveat: "bench_version 0.3 exports null latency fields.",
    label: "ITL",
    status: "unavailable",
    summary: "Inter-token latency in milliseconds. Not measured in the current export.",
  },
  {
    caveat: "Some generative exact-match rows can undercount valid formatted answers.",
    label: "Accuracy",
    status: "directional",
    summary: "Task score from committed evaluation artifacts. Higher means better task performance.",
  },
  {
    caveat: "MTP-on divided by autoregressive baseline for matching scenario pairs.",
    label: "MTPLX speedup",
    status: "measured",
    summary: "Relative generation speedup for speculative decoding comparisons.",
  },
];

export function MetricGlossary() {
  return (
    <div className="table-scroll" role="region" aria-label="Metric glossary" tabIndex={0}>
      <table className="data-table glossary-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Status</th>
            <th>Interpretation</th>
            <th>Caveat</th>
          </tr>
        </thead>
        <tbody>
          {metricGlossary.map((item) => (
            <tr key={item.label}>
              <td>
                <strong>{item.label}</strong>
              </td>
              <td>
                <span className={`badge ${item.status}`}>{item.status}</span>
              </td>
              <td>{item.summary}</td>
              <td className="muted">{item.caveat}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
