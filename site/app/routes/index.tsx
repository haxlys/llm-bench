import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: SummaryRoute,
});

function SummaryRoute() {
  return <section aria-label="Benchmark summary" />;
}
