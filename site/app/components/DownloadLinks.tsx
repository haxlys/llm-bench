import { Download } from "lucide-react";

type DownloadArtifact = {
  description: string;
  href: string;
  label: string;
};

const artifacts: DownloadArtifact[] = [
  {
    description: "Typed site export used by the report pages.",
    href: "/data/benchmarks.json",
    label: "benchmarks.json",
  },
  {
    description: "Combined benchmark speed summary.",
    href: "/data/summary.csv",
    label: "summary.csv",
  },
  {
    description: "Primary evaluation score summary.",
    href: "/data/eval_summary_primary.csv",
    label: "eval_summary_primary.csv",
  },
  {
    description: "MTPLX speculative decoding speedup rows.",
    href: "/data/mtplx_speedups.csv",
    label: "mtplx_speedups.csv",
  },
];

export function DownloadLinks() {
  return (
    <div className="download-grid">
      {artifacts.map((artifact) => (
        <a className="download-link panel" href={artifact.href} key={artifact.href}>
          <Download size={18} aria-hidden="true" />
          <span>
            <strong>{artifact.label}</strong>
            <small>{artifact.description}</small>
          </span>
        </a>
      ))}
    </div>
  );
}
