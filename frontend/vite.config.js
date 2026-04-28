import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    modulePreload: {
      resolveDependencies(_filename, deps) {
        return deps.filter((dep) => !/assets\/(charts|result-view)-/.test(dep));
      },
    },
  },
  worker: {
    format: "es",
  },
  assetsInclude: ["**/*.py"],
});
