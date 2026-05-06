import { HeadContent, Outlet, Scripts, createRootRoute } from "@tanstack/react-router";
import { Gauge, LineChart, ListChecks } from "lucide-react";

import "../styles.css";

const navItems = [
  { to: "/", label: "Summary", icon: LineChart },
  { to: "/accuracy", label: "Accuracy", icon: ListChecks },
  { to: "/speed", label: "Speed", icon: Gauge },
] as const;

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "llm-bench" },
    ],
  }),
  component: RootLayout,
});

function RootLayout() {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        <AppShell />
        <Scripts />
      </body>
    </html>
  );
}

function AppShell() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <a href="/" className="brand" aria-label="llm-bench summary">
          <span className="brand-mark">lb</span>
          <span>
            <strong>llm-bench</strong>
            <small>Apple Silicon local model benchmarks</small>
          </span>
        </a>
        <nav className="site-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <a key={item.to} href={item.to} className="nav-link">
                <Icon size={16} aria-hidden="true" />
                <span>{item.label}</span>
              </a>
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
