import type { ReactNode } from "react";

import type { MetricStatus } from "../lib/format";
import { statusLabel } from "../lib/format";

type BadgeProps = {
  status?: MetricStatus;
  children?: ReactNode;
};

export function Badge({ status, children }: BadgeProps) {
  const className = status ? `badge ${status}` : "badge";
  return <span className={className}>{children ?? statusLabel(status ?? "measured")}</span>;
}
