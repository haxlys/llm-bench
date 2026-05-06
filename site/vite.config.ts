import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => ({
  plugins: [
    ...(mode === "test" ? [] : [cloudflare({ viteEnvironment: { name: "ssr" } })]),
    tanstackStart({
      srcDirectory: "app",
      prerender: {
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
