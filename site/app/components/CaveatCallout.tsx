import type { ReactNode } from "react";

import { AlertTriangle } from "lucide-react";

type CaveatCalloutProps = {
  children: ReactNode;
  title: string;
};

export function CaveatCallout({ children, title }: CaveatCalloutProps) {
  return (
    <section className="panel caveat-callout">
      <AlertTriangle size={18} aria-hidden="true" />
      <div>
        <h2>{title}</h2>
        <div>{children}</div>
      </div>
    </section>
  );
}
