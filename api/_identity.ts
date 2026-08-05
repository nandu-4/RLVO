/**
 * Identity and the two operating modes.
 *
 * demo      — no valid session. Verification runs, nothing is stored, and every persistence-backed
 *             feature reports why it is unavailable instead of failing at the point of use.
 * workspace — a Supabase session was presented and verified. Data is scoped to that user, and the
 *             audit trail records a real name and email rather than a self-declared string.
 *
 * This replaces the anonymous workspace-token scheme. That scheme made tenancy depend on a bearer
 * secret held in localStorage: losing it lost the data, sharing it shared everything, and the
 * audit trail could only record who *said* they made a decision. A verified session fixes all
 * three, which is what an audit trail has to mean.
 *
 * The token is verified against Supabase's auth server on every request. It is never decoded and
 * trusted locally — an unverified JWT is just a string the client wrote.
 */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const ANON_KEY = process.env.SUPABASE_ANON_KEY || SERVICE_KEY;

export interface Identity {
  /** Supabase auth user id — the tenancy key for every stored row. */
  userId: string;
  email: string;
  name: string;
  avatarUrl: string | null;
}

export type Mode = "demo" | "workspace";

export function persistenceConfigured(): boolean {
  return Boolean(SUPABASE_URL && SERVICE_KEY);
}

export function bearerToken(req: any): string | null {
  const header = String(req.headers?.authorization || req.headers?.Authorization || "");
  const match = /^Bearer\s+(.+)$/i.exec(header.trim());
  const token = match?.[1]?.trim();
  return token && token.length > 20 ? token : null;
}

/** Service-role REST call. Only reached after an identity has been verified. */
export async function supabaseRest(path: string, init: RequestInit = {}): Promise<Response> {
  if (!persistenceConfigured()) throw new Error("Persistence is not configured on this deployment.");
  return fetch(`${SUPABASE_URL}/rest/v1/${path}`, {
    ...init,
    headers: {
      apikey: SERVICE_KEY as string,
      Authorization: `Bearer ${SERVICE_KEY}`,
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });
}

export async function restJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await supabaseRest(path, init);
  if (!response.ok) throw new Error(`Supabase ${response.status}: ${(await response.text()).slice(0, 300)}`);
  return (await response.json()) as T;
}

/**
 * Verify the caller's session. Returns null for a guest — never throws for "not signed in",
 * because demo mode is a supported state, not an error.
 */
export async function resolveIdentity(req: any): Promise<Identity | null> {
  if (!persistenceConfigured()) return null;
  const token = bearerToken(req);
  if (!token) return null;

  const response = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
    headers: { apikey: ANON_KEY as string, Authorization: `Bearer ${token}` },
  });
  if (!response.ok) return null;

  const user = (await response.json()) as {
    id?: string;
    email?: string;
    user_metadata?: Record<string, unknown>;
  };
  if (!user?.id) return null;

  const meta = user.user_metadata ?? {};
  return {
    userId: user.id,
    email: user.email ?? "",
    name:
      (meta.full_name as string) ||
      (meta.name as string) ||
      (user.email ?? "").split("@")[0] ||
      "Reviewer",
    avatarUrl: (meta.avatar_url as string) ?? (meta.picture as string) ?? null,
  };
}

/** Confirm a document belongs to the caller before any service-role write touches it. */
export async function assertDocumentOwned(documentId: string, identity: Identity): Promise<void> {
  if (!isUuid(documentId)) throw httpError(400, "documentId must be a persisted UUID.");
  const rows = await restJson<Array<{ id: string; user_id: string }>>(
    `documents?id=eq.${documentId}&select=id,user_id&limit=1`,
  );
  if (!rows[0]) throw httpError(404, "Document not found.");
  if (rows[0].user_id !== identity.userId) throw httpError(403, "This document belongs to another account.");
}

/** Append an API activity record. Never allowed to fail the request it describes. */
export async function logActivity(
  identity: Identity | null,
  entry: { route: string; action: string; statusCode: number; detail?: string; durationMs?: number },
): Promise<void> {
  if (!identity || !persistenceConfigured()) return;
  try {
    await supabaseRest("api_activity", {
      method: "POST",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({
        user_id: identity.userId,
        route: entry.route,
        action: entry.action,
        status_code: entry.statusCode,
        detail: entry.detail?.slice(0, 500) ?? null,
        duration_ms: entry.durationMs ?? null,
      }),
    });
  } catch {
    /* activity logging is best-effort by design */
  }
}

/** Reason shown wherever a persistence-backed feature is unavailable. */
export function demoModeReason(): string {
  return persistenceConfigured()
    ? "You are using TruthLens in demo mode. Verification works and nothing is stored — sign in with Google to save results, use the review queue, and see your dashboard."
    : "This deployment has no database configured, so results cannot be stored. Verification still works.";
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const isUuid = (value: unknown): value is string => typeof value === "string" && UUID_RE.test(value);

export function httpError(status: number, message: string): Error & { status: number } {
  return Object.assign(new Error(message), { status });
}

export function statusOf(error: unknown, fallback: number): number {
  const status = (error as { status?: unknown })?.status;
  return typeof status === "number" ? status : fallback;
}
