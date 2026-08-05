/**
 * Anonymous workspace resolution — tenancy without accounts.
 *
 * There is no sign-up. The browser mints a high-entropy token on first use and sends it as
 * `x-truthlens-workspace`. The server stores only its SHA-256 hash and scopes every query by the
 * resolved workspace id.
 *
 * The token is a bearer secret, like an unlisted share link: whoever holds it has the workspace,
 * there is no second factor, and it cannot be recovered. That is an acceptable trade for open
 * self-serve use and is NOT sufficient for regulated personal or health data. The UI says so too
 * rather than leaving the user to infer it.
 *
 * Two deployment modes, decided by environment and never by the client:
 *   stateless — SUPABASE_* unset. Verification runs, nothing is stored, review is unavailable.
 *   workspace — SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY set. Results persist per workspace.
 */
import { createHash } from "node:crypto";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

/** Tokens are opaque to us; we only require enough entropy that they cannot be guessed. */
const TOKEN_RE = /^[A-Za-z0-9_-]{24,128}$/;

export interface Workspace {
  id: string;
  name: string;
  retentionDays: number;
}

export function persistenceConfigured(): boolean {
  return Boolean(SUPABASE_URL && SERVICE_KEY);
}

export function workspaceToken(req: any): string | null {
  const raw = req.headers?.["x-truthlens-workspace"];
  const token = String(Array.isArray(raw) ? raw[0] : raw || "").trim();
  return TOKEN_RE.test(token) ? token : null;
}

const hashToken = (token: string) => createHash("sha256").update(token).digest("hex");

/** Service-role REST call. The browser never talks to the database directly. */
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
 * Resolve the caller's workspace, creating it on first sight of a token.
 *
 * Auto-provisioning is what makes "usable by anyone" real: the first verification a visitor runs
 * silently creates their workspace, and human review works from that moment on.
 */
export async function resolveWorkspace(req: any, options: { create?: boolean } = {}): Promise<Workspace | null> {
  if (!persistenceConfigured()) return null;
  const token = workspaceToken(req);
  if (!token) return null;

  const tokenHash = hashToken(token);
  const existing = await restJson<Array<{ id: string; name: string; retention_days: number }>>(
    `organizations?token_hash=eq.${tokenHash}&select=id,name,retention_days&limit=1`,
  );

  if (existing[0]) {
    // Fire-and-forget: a failed heartbeat must never fail the request it rode in on.
    void supabaseRest(`organizations?id=eq.${existing[0].id}`, {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({ last_seen_at: new Date().toISOString() }),
    }).catch(() => undefined);
    return { id: existing[0].id, name: existing[0].name, retentionDays: existing[0].retention_days };
  }

  if (options.create === false) return null;

  // A first-time visitor fires several requests at once — the batch page creates a job while
  // documents upload. Two concurrent inserts would collide on the token_hash unique index, so
  // upsert on conflict and read back whichever row won the race.
  const response = await supabaseRest("organizations?on_conflict=token_hash", {
    method: "POST",
    headers: { Prefer: "return=representation,resolution=merge-duplicates" },
    body: JSON.stringify([{ name: "Personal workspace", token_hash: tokenHash, retention_days: defaultRetentionDays() }]),
  });

  if (response.ok) {
    const rows = (await response.json()) as Array<{ id: string; name: string; retention_days: number }>;
    if (rows[0]) return { id: rows[0].id, name: rows[0].name, retentionDays: rows[0].retention_days };
  }

  // Upsert returned no row: the winner's record is committed by now, so read it.
  const settled = await restJson<Array<{ id: string; name: string; retention_days: number }>>(
    `organizations?token_hash=eq.${tokenHash}&select=id,name,retention_days&limit=1`,
  );
  if (settled[0]) return { id: settled[0].id, name: settled[0].name, retentionDays: settled[0].retention_days };
  throw new Error("Workspace could not be provisioned.");
}

function defaultRetentionDays(): number {
  const configured = Number(process.env.DEFAULT_RETENTION_DAYS);
  return Number.isFinite(configured) && configured >= 1 && configured <= 3650 ? Math.round(configured) : 30;
}

/** Confirm a document belongs to the caller's workspace before any write touches it. */
export async function assertDocumentInWorkspace(documentId: string, workspace: Workspace): Promise<void> {
  if (!isUuid(documentId)) throw httpError(400, "documentId must be a persisted UUID.");
  const rows = await restJson<Array<{ id: string; organization_id: string }>>(
    `documents?id=eq.${documentId}&select=id,organization_id&limit=1`,
  );
  if (!rows[0]) throw httpError(404, "Document not found.");
  if (rows[0].organization_id !== workspace.id) throw httpError(403, "Document belongs to another workspace.");
}

/** Append an API activity record. Never allowed to fail the request it describes. */
export async function logActivity(
  workspaceId: string | null,
  entry: { route: string; action: string; statusCode: number; detail?: string; durationMs?: number },
): Promise<void> {
  if (!workspaceId || !persistenceConfigured()) return;
  try {
    await supabaseRest("api_activity", {
      method: "POST",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({
        organization_id: workspaceId,
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

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const isUuid = (value: unknown): value is string => typeof value === "string" && UUID_RE.test(value);

export function httpError(status: number, message: string): Error & { status: number } {
  return Object.assign(new Error(message), { status });
}

export function statusOf(error: unknown, fallback: number): number {
  const status = (error as { status?: unknown })?.status;
  return typeof status === "number" ? status : fallback;
}

/** Message shown wherever persistence is unavailable, so the reason is never a mystery. */
export function unavailableReason(hasToken: boolean): string {
  if (!persistenceConfigured()) {
    return "This deployment runs in stateless mode; results are not stored and human review is unavailable.";
  }
  return hasToken
    ? "This workspace could not be resolved, so the result was not stored."
    : "No workspace token was sent, so the result was not stored.";
}
