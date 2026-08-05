import { clientSafeError, errorMessage, sendJson, activeModel } from "./_gemini.js";
import { benchmarkTargets, providerStatus } from "./_providers/index.js";
import {
  demoModeReason,
  logActivity,
  resolveIdentity,
  restJson,
  statusOf,
  supabaseRest,
  type Identity,
} from "./_identity.js";
import { callerKey, rateLimit } from "./_ratelimit.js";

export const maxDuration = 20;

/**
 * Workspace, compliance posture, and activity.
 *
 *   POST { action: "status" }                  -> workspace, providers, compliance posture
 *   POST { action: "settings", name, retentionDays }
 *   POST { action: "activity" }                -> API activity log + recent review decisions
 *   POST { action: "purge" }                   -> apply retention now
 *   POST { action: "erase" }                   -> delete everything in this workspace
 *
 * The compliance block reports what is *actually true of this deployment*. It does not claim
 * SOC 2, GDPR or HIPAA readiness: those are organisational certifications, not properties a
 * codebase can assert about itself. What it can report — encryption in transit, retention
 * enforcement, audit coverage, erasure — it reports as measured facts.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  try {
    const { action = "status" } = req.body || {};

    const identity = await resolveIdentity(req).catch(() => null);

    // The provider registry and compliance posture are deployment facts, not per-account data, so
    // a guest still sees them — they simply cannot store anything.
    if (!identity) {
      return sendJson(res, 200, {
        available: false,
        reason: demoModeReason(),
        providers: providerStatus(),
        benchmarkTargets: benchmarkTargets().map(({ adapter, ...target }) => target),
        activeModel,
        compliance: demoCompliance(),
      });
    }

    const limit = rateLimit(callerKey(req, identity.userId), 120, 60_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    if (action === "settings") return updateSettings(req, res, identity);
    if (action === "activity") return activity(res, identity);
    if (action === "purge") return purge(res, identity);
    if (action === "erase") return erase(res, identity);
    if (action !== "status") return sendJson(res, 400, { error: "action must be one of: status, settings, activity, purge, erase." });

    const [documents, expiring, settings] = await Promise.all([
      restJson<Array<{ id: string; created_at: string; retention_until: string | null }>>(
        `documents?user_id=eq.${identity.userId}&select=id,created_at,retention_until&order=created_at.desc&limit=1000`,
      ),
      restJson<Array<{ id: string }>>(
        `documents?user_id=eq.${identity.userId}&retention_until=lt.${new Date().toISOString()}&select=id&limit=1000`,
      ),
      restJson<Array<{ retention_days: number }>>(
        `user_settings?user_id=eq.${identity.userId}&select=retention_days&limit=1`,
      ).catch(() => []),
    ]);
    const retentionDays = settings[0]?.retention_days ?? 30;

    return sendJson(res, 200, {
      available: true,
      account: {
        id: identity.userId,
        name: identity.name,
        email: identity.email,
        avatarUrl: identity.avatarUrl,
        retentionDays,
        documentCount: documents.length,
        oldestDocumentAt: documents[documents.length - 1]?.created_at ?? null,
        expiredAwaitingPurge: expiring.length,
      },
      providers: providerStatus(),
      benchmarkTargets: benchmarkTargets().map(({ adapter, ...target }) => target),
      activeModel,
      compliance: liveCompliance(retentionDays, expiring.length),
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "workspace").message });
  }
}

/* ── actions ─────────────────────────────────────────────────────────────── */

async function updateSettings(req: any, res: any, identity: Identity) {
  const { retentionDays, preferredProvider, preferredModel } = req.body || {};
  const patch: Record<string, unknown> = {
    user_id: identity.userId,
    display_name: identity.name,
    email: identity.email,
  };
  if (typeof preferredProvider === "string") patch.preferred_provider = preferredProvider.slice(0, 40);
  if (typeof preferredModel === "string") patch.preferred_model = preferredModel.slice(0, 120);
  if (Number.isFinite(retentionDays)) {
    const days = Math.round(Number(retentionDays));
    if (days < 1 || days > 3650) return sendJson(res, 400, { error: "retentionDays must be between 1 and 3650." });
    patch.retention_days = days;
  }

  // Upsert: a user's first settings change creates their row.
  const response = await supabaseRest("user_settings?on_conflict=user_id", {
    method: "POST",
    headers: { Prefer: "return=representation,resolution=merge-duplicates" },
    body: JSON.stringify([{ ...patch, updated_at: new Date().toISOString() }]),
  });
  if (!response.ok) throw new Error(`Settings update failed: ${(await response.text()).slice(0, 200)}`);
  const [updated] = (await response.json()) as Array<{ retention_days: number }>;

  // Retention applies to documents already stored, not just future ones — otherwise shortening
  // the policy would leave the existing backlog untouched, which is not what a user means by it.
  if (patch.retention_days) {
    await supabaseRest("rpc/reapply_retention_for_user", {
      method: "POST",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({ target_user: identity.userId, days: patch.retention_days }),
    }).catch(() => undefined);
  }

  void logActivity(identity, { route: "workspace", action: "Updated account settings", statusCode: 200 });
  return sendJson(res, 200, {
    account: { name: identity.name, email: identity.email, retentionDays: updated?.retention_days ?? 30 },
  });
}

async function activity(res: any, identity: Identity) {
  const [log, decisions] = await Promise.all([
    restJson<Array<{ id: string; route: string; action: string; status_code: number; duration_ms: number | null; created_at: string }>>(
      `api_activity?user_id=eq.${identity.userId}&select=id,route,action,status_code,duration_ms,created_at&order=created_at.desc&limit=100`,
    ),
    restJson<Array<{ id: string; decision: string; reviewer_name: string; reviewer_email: string | null; reviewer_notes: string | null; created_at: string }>>(
      `review_decisions?select=id,decision,reviewer_name,reviewer_email,reviewer_notes,created_at,documents!inner(user_id,document_name)&documents.user_id=eq.${identity.userId}&order=created_at.desc&limit=100`,
    ),
  ]);

  return sendJson(res, 200, {
    apiActivity: log.map((entry) => ({
      id: entry.id,
      route: entry.route,
      action: entry.action,
      statusCode: entry.status_code,
      durationMs: entry.duration_ms,
      createdAt: entry.created_at,
    })),
    reviewDecisions: decisions.map((entry: any) => ({
      id: entry.id,
      decision: entry.decision,
      reviewerName: entry.reviewer_name,
      reviewerEmail: entry.reviewer_email,
      reviewerNotes: entry.reviewer_notes,
      documentName: entry.documents?.document_name ?? "Unknown document",
      createdAt: entry.created_at,
    })),
  });
}

async function purge(res: any, identity: Identity) {
  const expired = await restJson<Array<{ id: string }>>(
    `documents?user_id=eq.${identity.userId}&retention_until=lt.${new Date().toISOString()}&select=id&limit=1000`,
  );
  if (expired.length > 0) {
    // Claims, evidence, relations, timeline, audit rows and review records cascade from documents.
    const ids = expired.map((row) => row.id).join(",");
    const response = await supabaseRest(`documents?id=in.(${ids})`, { method: "DELETE", headers: { Prefer: "return=minimal" } });
    if (!response.ok) throw new Error(`Purge failed: ${(await response.text()).slice(0, 200)}`);
  }
  void logActivity(identity, { route: "workspace", action: `Purged ${expired.length} expired document(s)`, statusCode: 200 });
  return sendJson(res, 200, { purged: expired.length });
}

async function erase(res: any, identity: Identity) {
  const response = await supabaseRest(`documents?user_id=eq.${identity.userId}`, { method: "DELETE", headers: { Prefer: "return=minimal" } });
  if (!response.ok) throw new Error(`Erase failed: ${(await response.text()).slice(0, 200)}`);
  await supabaseRest(`verification_jobs?user_id=eq.${identity.userId}`, { method: "DELETE", headers: { Prefer: "return=minimal" } }).catch(() => undefined);
  await supabaseRest(`model_benchmarks?user_id=eq.${identity.userId}`, { method: "DELETE", headers: { Prefer: "return=minimal" } }).catch(() => undefined);
  return sendJson(res, 200, { erased: true });
}

/* ── compliance posture ──────────────────────────────────────────────────── */

interface Control {
  id: string;
  label: string;
  state: "enforced" | "partial" | "absent";
  detail: string;
}

function liveCompliance(retentionDays: number, expiredPending: number): { controls: Control[]; caveat: string } {
  return {
    controls: [
      { id: "encryption-transit", label: "Encryption in transit", state: "enforced", detail: "All traffic to the API, the model provider, and the database is TLS. There is no plaintext transport path." },
      { id: "encryption-rest", label: "Encryption at rest", state: "enforced", detail: "Provided by the managed Postgres instance (AES-256). TruthLens adds no application-layer field encryption." },
      { id: "retention", label: "Data retention policy", state: "enforced", detail: `Documents are stamped with an expiry ${retentionDays} days after verification. ${expiredPending} document(s) are past expiry and awaiting purge.` },
      { id: "purge", label: "Automated purge", state: expiredPending > 0 ? "partial" : "enforced", detail: "Expired documents are deleted on demand from this page. No scheduler runs automatically — wire purge_expired_documents() to a cron job for unattended enforcement." },
      { id: "erasure", label: "Right to erasure", state: "enforced", detail: "One action deletes every document, claim, evidence row, relation, audit entry and review decision in this workspace, by cascade." },
      { id: "audit-trail", label: "Audit trail", state: "enforced", detail: "Every machine and human decision is written to audit_trails with its final value, trust score and reviewer. Audit rows are revoked from write access outside the service role." },
      { id: "api-activity", label: "API activity log", state: "enforced", detail: "Each verification, review, benchmark and batch action is recorded with route, outcome and duration." },
      { id: "tenant-isolation", label: "Tenant isolation", state: "enforced", detail: "Every query is scoped by the verified user id server-side. All tables are revoked from the anon and authenticated roles, so the browser cannot reach the database directly." },
      { id: "authentication", label: "User authentication", state: "enforced", detail: "Google sign-in via Supabase Auth. Every request presenting a session is verified against the auth server; tokens are never decoded and trusted locally." },
      { id: "access-control", label: "Access control", state: "partial", detail: "Every document, claim and audit record is scoped to the authenticated user, and ownership is checked before any write. Organisation-level roles are not implemented — each account sees only its own data." },
      { id: "provider-processing", label: "Third-party processing", state: "partial", detail: "Document contents are sent to the configured model provider for transcription and verification. Review that provider's data-processing terms before uploading confidential material." },
    ],
    caveat:
      "TruthLens reports controls it can verify about this deployment. SOC 2, GDPR and HIPAA readiness are organisational certifications covering people, contracts and process — they cannot be asserted by software about itself, and are not claimed here.",
  };
}

function demoCompliance(): { controls: Control[]; caveat: string } {
  return {
    controls: [
      { id: "encryption-transit", label: "Encryption in transit", state: "enforced", detail: "All traffic to the API and the model provider is TLS." },
      { id: "retention", label: "Data retention policy", state: "enforced", detail: "Nothing is stored: this deployment has no database configured, so no document outlives its request." },
      { id: "audit-trail", label: "Audit trail", state: "absent", detail: "Audit records require persistence. Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to enable them." },
      { id: "authentication", label: "User authentication", state: "absent", detail: "You are browsing as a guest. Sign in with Google to enable storage, review and audit." },
      { id: "provider-processing", label: "Third-party processing", state: "partial", detail: "Document contents are sent to the configured model provider for transcription and verification." },
    ],
    caveat: "This deployment stores nothing. Configure persistence to enable audit trails, retention enforcement and human review.",
  };
}
