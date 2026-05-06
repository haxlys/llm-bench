import { createFileRoute } from "@tanstack/react-router";

import { SpeedPage } from "../pages/SpeedPage";

export const Route = createFileRoute("/speed")({
  component: SpeedRoute,
});

function SpeedRoute() {
  return <SpeedPage locale="en" />;
}
