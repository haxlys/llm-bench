import { createFileRoute } from "@tanstack/react-router";

import { AccuracyPage } from "../../pages/AccuracyPage";

export const Route = createFileRoute("/ko/accuracy")({
  component: AccuracyRoute,
});

function AccuracyRoute() {
  return <AccuracyPage locale="ko" />;
}
