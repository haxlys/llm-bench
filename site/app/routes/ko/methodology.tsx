import { createFileRoute } from "@tanstack/react-router";

import { MethodologyPage } from "../../pages/MethodologyPage";

export const Route = createFileRoute("/ko/methodology")({
  component: MethodologyRoute,
});

function MethodologyRoute() {
  return <MethodologyPage locale="ko" />;
}
