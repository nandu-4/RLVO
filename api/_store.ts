/**
 * Session storage abstraction: one interface, two drivers.
 *
 *   supabase — a verified Google session, Postgres, row-level tenancy, durable and shareable.
 *              Chosen whenever SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are both present.
 *   local    — files on disk keyed by a browser-held device id. No signup, no migrations, no
 *              sign-in. Chosen when Supabase is absent, unless TRUTHLENS_LOCAL_STORE=off.
 *
 * Callers never branch on the driver. `resolveSession` hands back whichever is live and the
 * functions below do the right thing, so verify-document and history contain no storage
 * conditionals beyond "is there a session at all".
 *
 * The drivers are NOT equivalent and the difference is deliberate. Supabase gives verified
 * identity, which is what makes an audit trail meaningful; the local driver gives availability,
 * which is what makes the feature exist for someone who has not configured anything. Anything
 * requiring attributable identity — the review queue, audit trails — stays on Supabase only, and
 * says so rather than quietly accepting a device id as a reviewer.
 */
import { persistenceConfigured, resolveIdentity, restJson, type Identity } from "./_identity.js";
import { persistReplay } from "./_persistence.js";
import {
  contentHash,
  deviceIdFrom,
  findByHash,
  listSessions,
  localStoreEnabled,
  readSession,
  saveSession,
  storageDurability,
  type SessionSummary,
} from "./_localstore.js";

export type StorageDriver = "supabase" | "local" | "none";

export interface StorageSession {
  driver: StorageDriver;
  /** Present only on the supabase driver. */
  identity?: Identity;
  /** Present only on the local driver. */
  deviceId?: string;
  /** True when stored data is attributable to a verified human. */
  attributable: boolean;
}

export { contentHash };

export function storageDriver(): StorageDriver {
  if (persistenceConfigured()) return "supabase";
  return localStoreEnabled() ? "local" : "none";
}

/**
 * Resolve where this request's results should be stored.
 *
 * Returns null only when there is genuinely nowhere to write. Note the asymmetry: on the Supabase
 * driver an unauthenticated caller gets null (demo mode is a real, supported state), whereas the
 * local driver always yields a session, because requiring sign-in to write a file to your own disk
 * would be ceremony without a purpose.
 */
export async function resolveSession(req: any): Promise<StorageSession | null> {
  if (persistenceConfigured()) {
    const identity = await resolveIdentity(req);
    return identity ? { driver: "supabase", identity, attributable: true } : null;
  }
  if (!localStoreEnabled()) return null;
  return { driver: "local", deviceId: deviceIdFrom(req), attributable: false };
}

/** Why storage is unavailable, phrased for the person reading it. */
export function storageUnavailableReason(): string {
  if (persistenceConfigured()) {
    return "You are using TruthLens in demo mode. Verification works and nothing is stored — sign in with Google to save results, use the review queue, and see your dashboard.";
  }
  if (!localStoreEnabled()) {
    return "Session storage is disabled on this deployment (TRUTHLENS_LOCAL_STORE=off), so results are not stored. Verification still works.";
  }
  return "Results could not be stored on this deployment.";
}

export type StoredSummary = SessionSummary;

/**
 * Persist the complete verification snapshot.
 *
 * `snapshot` is the entire response object the browser received — OCR-derived blocks, retrieved
 * evidence with coordinates, per-claim reasoning and trust breakdowns, the relation graph, risk
 * prediction, measured stage timings, provider, model and failover attempts. Storing the whole
 * envelope rather than a normalised subset is what allows a replay to be byte-identical to the
 * original run; a verdict-only row could never reconstruct the evidence overlay.
 */
export async function saveVerification(
  session: StorageSession,
  snapshot: Record<string, unknown>,
  summary: StoredSummary,
): Promise<void> {
  if (session.driver === "local") {
    await saveSession(session.deviceId as string, snapshot, summary);
    return;
  }
  // The hash rides inside the snapshot so repeat-detection needs no schema change.
  await persistReplay(summary.id, session.identity as Identity, snapshot, (snapshot as { attempts?: unknown }).attempts ?? []);
}

export async function listHistory(session: StorageSession): Promise<StoredSummary[]> {
  if (session.driver === "local") return listSessions(session.deviceId as string);

  const rows = await restJson<Array<{ document_id: string; created_at: string; verification_snapshot: any }>>(
    `verification_replays?user_id=eq.${session.identity!.userId}&select=document_id,created_at,verification_snapshot&order=created_at.desc&limit=100`,
  );
  return rows.map((row) => ({
    id: row.document_id,
    createdAt: row.created_at,
    fileName: row.verification_snapshot?.fileName ?? "document",
    documentType: row.verification_snapshot?.documentType ?? "Unknown",
    provider: row.verification_snapshot?.providerLabel || row.verification_snapshot?.provider || "unknown",
    model: row.verification_snapshot?.modelUsed ?? "unknown",
    trustScore: row.verification_snapshot?.summary?.trustScore ?? 0,
    verificationMode: row.verification_snapshot?.verificationMode ?? "cross-check",
    contentHash: row.verification_snapshot?.contentHash,
    totalClaims: row.verification_snapshot?.summary?.totalClaims,
  }));
}

export async function readReplay(session: StorageSession, id: string): Promise<Record<string, unknown> | null> {
  if (session.driver === "local") return readSession(session.deviceId as string, id);

  const rows = await restJson<Array<{ verification_snapshot: Record<string, unknown> }>>(
    `verification_replays?document_id=eq.${id}&user_id=eq.${session.identity!.userId}&select=verification_snapshot&limit=1`,
  );
  return rows[0]?.verification_snapshot ?? null;
}

/**
 * Find a previous run of the identical document and claims, so it can be returned without
 * spending a model call.
 *
 * `requestedProvider` guards against a silent substitution: if the caller explicitly asked for a
 * provider or model, a cached answer from a different one is not what they asked for, so the
 * cache is skipped and the model runs. Without that check, choosing a model in Admin → Models
 * would appear to do nothing.
 */
export async function findRepeat(
  session: StorageSession,
  hash: string,
  requestedProvider?: string,
  requestedModel?: string,
): Promise<Record<string, unknown> | null> {
  let match: { id: string; provider: string; model: string } | null = null;

  if (session.driver === "local") {
    const found = await findByHash(session.deviceId as string, hash);
    if (found) match = { id: found.id, provider: found.provider, model: found.model };
  } else {
    const rows = await restJson<Array<{ document_id: string; verification_snapshot: any }>>(
      `verification_replays?user_id=eq.${session.identity!.userId}&verification_snapshot->>contentHash=eq.${hash}&select=document_id,verification_snapshot&order=created_at.desc&limit=1`,
    );
    if (rows[0]) {
      match = {
        id: rows[0].document_id,
        provider: rows[0].verification_snapshot?.provider ?? "",
        model: rows[0].verification_snapshot?.modelUsed ?? "",
      };
    }
  }

  if (!match) return null;
  if (requestedProvider && match.provider && requestedProvider !== match.provider) return null;
  if (requestedModel && match.model && requestedModel !== match.model) return null;

  return readReplay(session, match.id);
}

export { storageDurability };
