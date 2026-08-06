/**
 * Zero-configuration local session store.
 *
 * WHY THIS EXISTS
 * The full replay architecture — complete snapshots, history listing, no-model replay — was
 * already built, and every part of it was unreachable: it required a Supabase project, a service
 * role key, five applied migrations and a Google sign-in before a single result could be stored.
 * A deployment with none of those reported "no database configured, so results cannot be stored"
 * and silently discarded every verification. This backend makes storage the default rather than
 * a reward for finishing a setup checklist.
 *
 * WHAT IT IS NOT
 * It is not a database and does not pretend to be one. There are no transactions, no
 * cross-process locking beyond atomic file replacement, and no query engine. It stores whole
 * session documents keyed by device, which is exactly the shape replay needs and nothing more.
 * When SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set, the Supabase driver takes over
 * completely and this code is never consulted — see `storageDriver()` in _store.ts.
 *
 * DURABILITY, STATED HONESTLY
 * Local disk is durable when the process owns a real filesystem: `vercel dev`, `node`, a
 * container, a VM. On Vercel's managed runtime only /tmp is writable and it is per-instance and
 * evicted freely, so sessions there survive minutes, not days. `storageDurability()` reports
 * which of those two situations is live so the UI can say so rather than imply permanence it
 * cannot deliver.
 */
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, writeFile, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

export interface SessionSummary {
  id: string;
  createdAt: string;
  fileName: string;
  documentType: string;
  provider: string;
  model: string;
  trustScore: number;
  verificationMode: string;
  /** Content address of document + claims; the key for a no-model repeat verification. */
  contentHash?: string;
  totalClaims?: number;
}

const MAX_SESSIONS_PER_DEVICE = 200;
const DEVICE_HEADER = "x-truthlens-device";
/** Device ids come from a browser and are used to build paths — treat them as hostile input. */
const SAFE_ID = /^[A-Za-z0-9_-]{8,64}$/;

/** Explicit opt-out for deployments that genuinely want statelessness. */
export function localStoreEnabled(): boolean {
  return process.env.TRUTHLENS_LOCAL_STORE !== "off";
}

function root(): string {
  const configured = process.env.TRUTHLENS_DATA_DIR;
  if (configured) return resolve(configured);
  // Vercel's managed runtime has a read-only bundle; /tmp is the only writable path.
  if (process.env.VERCEL) return join(tmpdir(), "truthlens-data");
  return resolve(process.cwd(), ".truthlens-data");
}

export function storageDurability(): "durable" | "ephemeral" {
  return process.env.VERCEL && !process.env.TRUTHLENS_DATA_DIR ? "ephemeral" : "durable";
}

/**
 * Stable per-browser identifier. Not a credential and not a security boundary: it separates one
 * browser's history from another's on a shared deployment, nothing more. Anyone who can send the
 * header can read that device's sessions, which is why the Supabase driver — with a verified
 * session behind it — remains the only path recommended for shared or sensitive deployments.
 */
export function deviceIdFrom(req: any): string {
  const raw = String(req?.headers?.[DEVICE_HEADER] ?? req?.headers?.[DEVICE_HEADER.toUpperCase()] ?? "").trim();
  return SAFE_ID.test(raw) ? raw : "shared-local";
}

const deviceDir = (deviceId: string) => join(root(), SAFE_ID.test(deviceId) ? deviceId : "shared-local");
const indexPath = (deviceId: string) => join(deviceDir(deviceId), "index.json");
const sessionPath = (deviceId: string, id: string) => join(deviceDir(deviceId), `${id}.json`);

/**
 * Content address for a verification request.
 *
 * Deliberately covers only what determines the answer — the document bytes, the claims asked
 * about, and the mode. Provider and model are excluded so that a result produced by a failover
 * model still satisfies a later identical request; the caller decides whether that substitution
 * is acceptable and `_store.ts` refuses the cache when a specific provider was demanded.
 */
export function contentHash(documentData: string, claims: Array<{ field: string; value: string }>, mode: string): string {
  /*
   * JSON-encode the pairs instead of joining them with a separator character.
   *
   * Building each entry as `field=value` let a claim smuggle the delimiter:
   * {field:"a=b", value:"c"} and {field:"a", value:"b=c"} both flattened to "a=b=c", so two
   * genuinely different questions collided on one cache entry and the second would have been
   * answered with the first one's stored result. JSON escaping makes the encoding injective,
   * which is the property a cache key has to have.
   */
  const normalised = JSON.stringify(
    claims
      .map((c) => [c.field.trim().toLowerCase(), c.value.trim().toLowerCase()])
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0]))),
  );
  return createHash("sha256")
    .update(JSON.stringify({ mode, normalised, documentLength: documentData.length }))
    .update(documentData)
    .digest("hex");
}

async function readIndex(deviceId: string): Promise<SessionSummary[]> {
  try {
    const raw = await readFile(indexPath(deviceId), "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // A missing or corrupt index must not take the endpoint down; an empty history is recoverable,
    // a 500 on every page load is not.
    return [];
  }
}

/** Atomic replace: a crash mid-write leaves the previous index intact rather than a truncated one. */
async function writeAtomic(path: string, contents: string): Promise<void> {
  const temp = `${path}.${randomUUID()}.tmp`;
  await writeFile(temp, contents, "utf8");
  await rename(temp, path);
}

export async function saveSession(deviceId: string, snapshot: Record<string, unknown>, summary: SessionSummary): Promise<void> {
  const dir = deviceDir(deviceId);
  if (!existsSync(dir)) await mkdir(dir, { recursive: true });

  await writeAtomic(sessionPath(deviceId, summary.id), JSON.stringify(snapshot));

  const index = await readIndex(deviceId);
  const next = [summary, ...index.filter((s) => s.id !== summary.id)];

  // Bound the store so a long-running deployment cannot fill the disk.
  const evicted = next.slice(MAX_SESSIONS_PER_DEVICE);
  await Promise.all(
    evicted.map((s) => rm(sessionPath(deviceId, s.id), { force: true }).catch(() => undefined)),
  );
  await writeAtomic(indexPath(deviceId), JSON.stringify(next.slice(0, MAX_SESSIONS_PER_DEVICE)));
}

export async function listSessions(deviceId: string): Promise<SessionSummary[]> {
  return readIndex(deviceId);
}

export async function readSession(deviceId: string, id: string): Promise<Record<string, unknown> | null> {
  if (!SAFE_ID.test(id) && !/^[0-9a-f-]{36}$/i.test(id)) return null;
  try {
    return JSON.parse(await readFile(sessionPath(deviceId, id), "utf8"));
  } catch {
    return null;
  }
}

/** Most recent session whose inputs hash identically. */
export async function findByHash(deviceId: string, hash: string): Promise<SessionSummary | null> {
  const index = await readIndex(deviceId);
  return index.find((s) => s.contentHash === hash) ?? null;
}

export async function deleteSession(deviceId: string, id: string): Promise<boolean> {
  const index = await readIndex(deviceId);
  if (!index.some((s) => s.id === id)) return false;
  await rm(sessionPath(deviceId, id), { force: true }).catch(() => undefined);
  await writeAtomic(indexPath(deviceId), JSON.stringify(index.filter((s) => s.id !== id)));
  return true;
}

/** Diagnostics for /api/health. Never throws — health must report, not fail. */
export async function localStoreStatus(): Promise<{ ok: boolean; path: string; devices: number; sessions: number; durability: string; detail: string }> {
  const base = root();
  const durability = storageDurability();
  try {
    if (!existsSync(base)) {
      return { ok: true, path: base, devices: 0, sessions: 0, durability, detail: "Local store ready; nothing written yet." };
    }
    const devices = (await readdir(base, { withFileTypes: true })).filter((e) => e.isDirectory());
    let sessions = 0;
    for (const device of devices) sessions += (await readIndex(device.name)).length;
    return { ok: true, path: base, devices: devices.length, sessions, durability, detail: `${sessions} stored session(s) across ${devices.length} device(s).` };
  } catch (error) {
    return { ok: false, path: base, devices: 0, sessions: 0, durability, detail: error instanceof Error ? error.message : "Local store unreadable." };
  }
}
