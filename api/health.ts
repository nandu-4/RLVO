import { activeModel, sendJson } from "./_gemini.js";
import { persistenceConfigured, supabaseRest } from "./_identity.js";
import { providerStatus } from "./_providers/index.js";

export const maxDuration = 10;

/**
 * Deployment health check for uptime monitors and load balancers.
 *
 * Deliberately reports capability, never secrets: which subsystems are configured and reachable,
 * not their credentials or URLs. A monitor can alert on `status`, and an engineer can see at a
 * glance which half of the platform is degraded without opening a dashboard.
 *
 * Returns 200 when usable (even if degraded) and 503 only when verification cannot run at all,
 * so a monitor pages on genuine outage rather than on optional persistence being absent.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "GET" && req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });
  const startedAt = Date.now();

  const providers = providerStatus();
  const modelConfigured = providers.some((provider) => provider.configured);

  let database: { configured: boolean; reachable: boolean; latencyMs: number | null } = {
    configured: persistenceConfigured(),
    reachable: false,
    latencyMs: null,
  };

  if (database.configured) {
    const probeStartedAt = Date.now();
    try {
      // Cheapest possible round trip that still proves PostgREST and Postgres are both alive.
      const response = await supabaseRest("documents?select=id&limit=1");
      database = { configured: true, reachable: response.ok, latencyMs: Date.now() - probeStartedAt };
    } catch {
      database = { configured: true, reachable: false, latencyMs: Date.now() - probeStartedAt };
    }
  }

  const status = !modelConfigured
    ? "unhealthy" // verification is impossible
    : database.configured && !database.reachable
    ? "degraded" // verification works; storage, review and analytics do not
    : "healthy";

  res.setHeader("Cache-Control", "no-store");
  return sendJson(res, status === "unhealthy" ? 503 : 200, {
    status,
    version: process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? "dev",
    uptimeSeconds: Math.round(process.uptime?.() ?? 0),
    checks: {
      verification: { ok: modelConfigured, detail: modelConfigured ? "A vision provider adapter is configured." : "No provider adapter is configured; set GEMINI_API_KEY." },
      persistence: {
        ok: !database.configured || database.reachable,
        mode: database.configured ? "workspace" : "demo-only",
        latencyMs: database.latencyMs,
        detail: !database.configured
          ? "Stateless mode: results are not stored; review, analytics and audit are unavailable by design."
          : database.reachable
          ? "Database reachable."
          : "Database configured but unreachable — verification still works, storage does not.",
      },
    },
    activeModel,
    providers: providers.map(({ id, configured }) => ({ id, configured })),
    checkedInMs: Date.now() - startedAt,
    timestamp: new Date().toISOString(),
  });
}
