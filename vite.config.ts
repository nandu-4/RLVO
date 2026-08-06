import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * Dev-only API server.
 *
 * `npm run dev` runs plain Vite, which has no idea how to serve the Vercel `/api/*` functions —
 * every POST to /api/verify-document or /api/extract-claims returned 404 (the SPA fallback).
 * This middleware loads the real `api/*.ts` handlers through Vite's SSR transform pipeline and
 * invokes them with the same Node req/res surface Vercel provides, so the frontend and backend
 * work together under `npm run dev`. Production deployments are unchanged.
 *
 * Provider failover is the same in dev as in production: api/_providers/index.ts builds the
 * resolution chain from every configured provider in priority order gemini → openrouter →
 * huggingface (Qwen), and api/verify-document.ts falls through the chain whenever the current
 * provider errors (e.g. an exhausted Gemini quota). With only HUGGINGFACE_API_KEY set, Qwen
 * handles every verification directly.
 */
function apiRoutesDev(): Plugin {
  return {
    name: "truthlens-api-routes",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? "/", "http://localhost");
        if (!url.pathname.startsWith("/api/")) return next();

        /*
         * Mirror the production rewrites in vercel.json. The four RLVO endpoints are served by a
         * single consolidated function (Vercel's Hobby plan caps a deployment at 12 functions), so
         * their public URLs resolve to /api/rlvo with an `action`. Without this map, dev would 404
         * on routes that work in production — the worst kind of environment drift.
         */
        const RLVO_ACTIONS = new Set(["generate-caption", "refine-caption", "analyze-video", "verify-flag"]);

        // Map /api/name -> api/name.ts. Files starting with "_" are internal modules, not routes.
        let name = url.pathname.slice("/api/".length).split("/")[0];
        // The dev shim provides no req.query, so the dispatcher resolves the action from the
        // trailing path segment of req.url instead — which is still /api/<action> here.
        if (RLVO_ACTIONS.has(name)) name = "rlvo";
        if (!name || name.startsWith("_") || /[^a-z0-9-]/.test(name)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: `No API route named "${name}".` }));
          return;
        }

        try {
          const mod = await server.ssrLoadModule(`/api/${name}.ts`);
          if (typeof mod.default !== "function") {
            throw new Error(`/api/${name}.ts has no default handler export.`);
          }

          // Vercel delivers a parsed JSON body; reconstruct it from the request stream.
          const chunks: Buffer[] = [];
          for await (const chunk of req) {
            chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string));
          }
          const raw = Buffer.concat(chunks).toString("utf8").trim();
          if (raw) {
            try {
              (req as { body?: unknown }).body = JSON.parse(raw);
            } catch {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Request body must be valid JSON." }));
              return;
            }
          } else {
            (req as { body?: unknown }).body = {};
          }

          await mod.default(req, res);
        } catch (error) {
          console.error("[api-dev]", error);
          if (!res.headersSent) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: error instanceof Error ? error.message : "API route failed." }));
          }
        }
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // `npm run dev` is a Vite process: it only injects VITE_* vars into the browser bundle and
  // does NOT populate process.env from .env for server-side code. The API handlers read
  // process.env.GEMINI_API_KEY / OPENROUTER_API_KEY / HUGGINGFACE_API_KEY (and SUPABASE_*),
  // so without this the dev API would report "No AI provider is configured" even when .env
  // has a key — and the Qwen fallback would never be reached. Load the same file Vercel uses.
  const env = loadEnv(mode, process.cwd(), "");
  for (const [key, value] of Object.entries(env)) {
    if (process.env[key] === undefined) process.env[key] = value;
  }

  return {
    server: {
      host: "::",
      port: 8080,
    },
    plugins: [react(), apiRoutesDev()],
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
  };
});
