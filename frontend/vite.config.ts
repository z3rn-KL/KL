import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Dev server proxies /api to the FastAPI backend (default http://127.0.0.1:8011).
// Override with VITE_API_TARGET=http://host:port npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_TARGET || "http://127.0.0.1:8011";
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
        "cn": path.resolve(__dirname, "./src/lib/utils.ts"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5176,
      proxy: { "/api": { target, changeOrigin: true } },
    },
    preview: {
      host: "0.0.0.0",
      port: 5176,
      proxy: { "/api": { target, changeOrigin: true } },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ["react", "react-dom", "react-router-dom"],
            leaflet: ["leaflet", "react-leaflet"],
            charts: ["recharts"],
          },
        },
      },
    },
  };
});
