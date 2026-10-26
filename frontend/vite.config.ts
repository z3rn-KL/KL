import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Dev server proxies /api to the FastAPI backend (default http://localhost:8000).
// Override with VITE_API_TARGET=http://host:port npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_TARGET || "http://localhost:8000";
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
        "cn": path.resolve(__dirname, "./src/lib/utils.ts"),
      },
    },
    server: {
      port: 5173,
      proxy: { "/api": { target, changeOrigin: true } },
    },
    preview: {
      port: 5173,
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
