import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Unit tests for the pieces where being wrong is silent.
 *
 * Not a second end-to-end suite — the API tests already cover the contract, and
 * duplicating them here would only take longer to fail. What is tested is the
 * logic that lives in the browser and has no server equivalent: the pure
 * functions that decide what a reader sees.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    restoreMocks: true,
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
