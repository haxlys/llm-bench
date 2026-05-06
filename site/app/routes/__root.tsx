import {
  HeadContent,
  Outlet,
  Scripts,
  createRootRoute,
  useLocation,
} from "@tanstack/react-router";
import { BookOpen, Database, Gauge, LineChart, ListChecks } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import {
  localeFromPathname,
  localizedPath,
  messages,
  switchLocalePath,
  type BasePagePath,
  type NavKey,
} from "../lib/i18n";
import "../styles.css";

const navItems = [
  { key: "summary", to: "/", icon: LineChart },
  { key: "accuracy", to: "/accuracy", icon: ListChecks },
  { key: "speed", to: "/speed", icon: Gauge },
  { key: "methodology", to: "/methodology", icon: BookOpen },
  { key: "data", to: "/data", icon: Database },
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
  const pathname = useLocation({ select: (location) => location.pathname });
  const locale = localeFromPathname(pathname);

  return (
    <html lang={locale}>
      <head>
        <HeadContent />
      </head>
      <body>
        <AppShell locale={locale} pathname={pathname} />
        <Scripts />
      </body>
    </html>
  );
}

type AppShellProps = {
  locale: "en" | "ko";
  pathname: string;
};

function AppShell({ locale, pathname }: AppShellProps) {
  const t = messages[locale];
  const activeBasePath = switchLocalePath(pathname, "en");

  return (
    <div className="app-shell">
      <header className="site-header">
        <a href={localizedPath("/", locale)} className="brand" aria-label={t.root.brandAria}>
          <span className="brand-mark">lb</span>
          <span>
            <strong>llm-bench</strong>
            <small>{t.root.brandSubtitle}</small>
          </span>
        </a>
        <div className="header-actions">
          <nav className="site-nav" aria-label={t.root.navAria}>
            {navItems.map((item) => (
              <NavLink
                active={activeBasePath === item.to}
                icon={item.icon}
                key={item.to}
                label={t.root.nav[item.key as NavKey]}
                to={localizedPath(item.to as BasePagePath, locale)}
              />
            ))}
          </nav>
          <nav className="language-switcher" aria-label={t.root.languageAria}>
            {(["en", "ko"] as const).map((targetLocale) => (
              <a
                aria-current={locale === targetLocale ? "page" : undefined}
                className={locale === targetLocale ? "language-link active" : "language-link"}
                href={switchLocalePath(pathname, targetLocale)}
                key={targetLocale}
              >
                {messages[locale].root.languageLabels[targetLocale]}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="site-main">
        <Outlet />
      </main>
    </div>
  );
}

type NavLinkProps = {
  active: boolean;
  icon: LucideIcon;
  label: string;
  to: string;
};

function NavLink({ active, icon: Icon, label, to }: NavLinkProps) {
  return (
    <a
      aria-current={active ? "page" : undefined}
      className={active ? "nav-link active" : "nav-link"}
      href={to}
    >
      <Icon size={16} aria-hidden="true" />
      <span>{label}</span>
    </a>
  );
}
