import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import { defineConfig } from "vitest/config";

const prerenderPages = [
  "/",
  "/accuracy",
  "/speed",
  "/methodology",
  "/data",
  "/ko",
  "/ko/accuracy",
  "/ko/speed",
  "/ko/methodology",
  "/ko/data",
].map((path) => ({ path }));

export default defineConfig(({ mode }) => ({
  plugins: [
    ...(mode === "test" ? [] : [cloudflare({ viteEnvironment: { name: "ssr" } })]),
    tanstackStart({
      pages: prerenderPages,
      srcDirectory: "app",
      prerender: {
        autoStaticPathsDiscovery: false,
        crawlLinks: false,
        enabled: true,
      },
    }),
    react(),
  ],
  test: {
    environment: "jsdom",
    globals: true,
  },
}));
