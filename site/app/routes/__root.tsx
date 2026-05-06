import { Link, Outlet, createRootRoute } from "@tanstack/react-router";
import { Database, Gauge, LineChart, ListChecks, Microscope } from "lucide-react";

import "../styles.css";

const navItems = [
  { to: "/", label: "Summary", icon: LineChart },
  { to: "/accuracy", label: "Accuracy", icon: ListChecks },
  { to: "/speed", label: "Speed", icon: Gauge },
  { to: "/methodology", label: "Methodology", icon: Microscope },
  { to: "/data", label: "Data", icon: Database },
] as const;

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <Link to="/" className="brand" aria-label="llm-bench summary">
          <span className="brand-mark">lb</span>
          <span>
            <strong>llm-bench</strong>
            <small>Apple Silicon local model benchmarks</small>
          </span>
        </Link>
        <nav className="site-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link key={item.to} to={item.to} className="nav-link">
                <Icon size={16} aria-hidden="true" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </header>
      <main className="site-main">
        <Outlet />
      </main>
    </div>
  );
}
