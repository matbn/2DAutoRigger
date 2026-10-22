import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The editor is served by the Python service in production. During
// development the service runs separately and Vite proxies the API to it, so
// the browser still only ever talks to one origin.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../../../python/rigger2d/server/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
