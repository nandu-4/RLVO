import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    exclude: ["@mediapipe/face_mesh"],
  },
  build: {
    // Route chunks are already split via React.lazy in App.tsx. Separating the vendor libraries
    // as well means a deploy that only touches application code leaves these chunks cached —
    // the framework is the largest and least frequently changed part of the download.
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react": ["react", "react-dom", "react-router-dom"],
          "vendor-motion": ["framer-motion"],
          "vendor-query": ["@tanstack/react-query"],
        },
      },
    },
    // The pdf.js worker is legitimately ~1.3MB and loads only on the Evidence tab; the default
    // 500kB warning would fire on it every build and train us to ignore the warning entirely.
    chunkSizeWarningLimit: 700,
  },
});
