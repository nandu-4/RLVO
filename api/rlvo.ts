/**
 * RLVO Labs endpoints, served by one serverless function.
 *
 * Image RLVO, Video RLVO and Proctoring were four separate route files. Vercel's Hobby plan caps a
 * deployment at 12 Serverless Functions and the project had 14, so the deploy was rejected
 * outright. Consolidating the four smallest, most closely related handlers into a single dispatcher
 * brings the count to 11 without removing a single feature — the alternative was deleting working
 * tools to fit a quota, which is not a trade worth making.
 *
 * Public URLs are unchanged. `/api/generate-caption` and friends still work, via rewrites in
 * vercel.json for production and an alias map in vite.config.ts for `npm run dev`. Nothing in the
 * frontend needed to change, and the individual handlers are untouched in `_rlvo-*.ts` — files
 * prefixed with `_` are modules, not routes, so they cost no function slots.
 */
import { sendJson } from "./_gemini.js";
import generateCaption from "./_rlvo-generate-caption.js";
import refineCaption from "./_rlvo-refine-caption.js";
import analyzeVideo from "./_rlvo-analyze-video.js";
import verifyFlag from "./_rlvo-verify-flag.js";

/** The heaviest of the four decides the ceiling: caption generation reads a full image. */
export const maxDuration = 60;

type Handler = (req: any, res: any) => Promise<unknown> | unknown;

const ROUTES: Record<string, Handler> = {
  "generate-caption": generateCaption,
  "refine-caption": refineCaption,
  "analyze-video": analyzeVideo,
  "verify-flag": verifyFlag,
};

export default async function handler(req: any, res: any) {
  // The action arrives as a query parameter from the rewrite. Fall back to the trailing path
  // segment so a direct POST to /api/rlvo/verify-flag also resolves.
  const fromQuery = typeof req.query?.action === "string" ? req.query.action : "";
  const fromPath = String(req.url ?? "").split("?")[0].split("/").filter(Boolean).pop() ?? "";
  const action = ROUTES[fromQuery] ? fromQuery : fromPath;

  const route = ROUTES[action];
  if (!route) {
    return sendJson(res, 404, {
      error: `Unknown RLVO action "${action}". Expected one of: ${Object.keys(ROUTES).join(", ")}.`,
    });
  }
  return route(req, res);
}
