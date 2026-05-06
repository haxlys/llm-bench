import { createFileRoute } from "@tanstack/react-router";

import { SummaryPage } from "../../pages/SummaryPage";

export const Route = createFileRoute("/ko/")({
  component: SummaryRoute,
});

function SummaryRoute() {
  return <SummaryPage locale="ko" />;
}
