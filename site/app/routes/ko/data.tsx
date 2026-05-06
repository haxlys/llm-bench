import { createFileRoute } from "@tanstack/react-router";

import { DataPage } from "../../pages/DataPage";

export const Route = createFileRoute("/ko/data")({
  component: DataRoute,
});

function DataRoute() {
  return <DataPage locale="ko" />;
}
