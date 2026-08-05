import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import {
  demoModeReason,
  type Identity,
  httpError,
  isUuid,
  logActivity,
  resolveIdentity,
  restJson,
  statusOf,
  supabaseRest,
} from "./_identity.js";
import { callerKey, rateLimit } from "./_ratelimit.js";

export const maxDuration = 20;

const MAX_ITEMS = 1000;

/**
 * Batch verification job control.
 *
 * ARCHITECTURE NOTE — why the client drives the loop rather than the server:
 * a serverless function cannot hold a 1000-document queue open, and this deployment has no
 * worker or message broker. So the job record lives here and the browser submits documents one
 * at a time against it, reporting each outcome back. That gives real progress, real resumability
 * (the job survives a refresh), and a real consolidated report — without pretending there is a
 * distributed queue behind it. Moving to a true worker later means changing only this file and
 * the page that drives it; the job schema does not change.
 *
 *   POST { action: "create", files, label }  -> { job, items }
 *   POST { action: "item",   jobId, itemId, ... } -> records one item outcome
 *   POST { action: "status", jobId }         -> { job, items }
 *   POST { action: "cancel", jobId }
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  try {
    const identity = await resolveIdentity(req);
    if (!identity) return sendJson(res, 401, { error: demoModeReason() });

    const limit = rateLimit(callerKey(req, identity.userId), 240, 60_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const { action } = req.body || {};

    if (action === "create") return createJob(req, res, identity);
    if (action === "item") return recordItem(req, res, identity);
    if (action === "status") return jobStatus(req, res, identity);
    if (action === "cancel") return cancelJob(req, res, identity);
    return sendJson(res, 400, { error: "action must be one of: create, item, status, cancel." });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "batch").message });
  }
}

async function createJob(req: any, res: any, identity: Identity) {
  const { files, label } = req.body || {};
  if (!Array.isArray(files) || files.length === 0) {
    return sendJson(res, 400, { error: "files must be a non-empty array of file names." });
  }
  if (files.length > MAX_ITEMS) {
    return sendJson(res, 400, { error: `A job may contain at most ${MAX_ITEMS} documents.` });
  }

  const [job] = await write<Array<JobRow>>("verification_jobs", [{
    user_id: identity.userId,
    label: String(label || "Batch verification").slice(0, 120),
    status: "processing",
    total_documents: files.length,
  }]);

  const items = await write<Array<ItemRow>>(
    "verification_job_items",
    files.map((name: unknown, position: number) => ({
      job_id: job.id,
      file_name: String(name || `document-${position + 1}`).slice(0, 255),
      position,
      status: "queued",
    })),
  );

  void logActivity(identity, { route: "batch-job", action: `Created batch of ${files.length} document(s)`, statusCode: 201 });
  return sendJson(res, 201, { job: toJob(job), items: items.map(toItem) });
}

async function recordItem(req: any, res: any, identity: Identity) {
  const { jobId, itemId, status, documentId, trustScore, totalClaims, needsReviewClaims, errorDetail } = req.body || {};
  if (!isUuid(jobId) || !isUuid(itemId)) throw httpError(400, "jobId and itemId must be UUIDs.");
  if (!["processing", "completed", "failed"].includes(String(status))) {
    throw httpError(400, "status must be processing, completed, or failed.");
  }
  await assertJobOwned(jobId, identity.userId);

  const patch: Record<string, unknown> = { status };
  if (status !== "processing") patch.completed_at = new Date().toISOString();
  if (isUuid(documentId)) patch.document_id = documentId;
  if (Number.isFinite(trustScore)) patch.trust_score = clamp(Number(trustScore));
  if (Number.isFinite(totalClaims)) patch.total_claims = Math.max(0, Math.round(Number(totalClaims)));
  if (Number.isFinite(needsReviewClaims)) patch.needs_review_claims = Math.max(0, Math.round(Number(needsReviewClaims)));
  if (typeof errorDetail === "string") patch.error_detail = errorDetail.slice(0, 500);

  const patched = await supabaseRest(`verification_job_items?id=eq.${itemId}&job_id=eq.${jobId}`, {
    method: "PATCH",
    headers: { Prefer: "return=minimal" },
    body: JSON.stringify(patch),
  });
  if (!patched.ok) throw new Error(`Job item update failed: ${(await patched.text()).slice(0, 200)}`);

  return sendJson(res, 200, await recountJob(jobId));
}

async function jobStatus(req: any, res: any, identity: Identity) {
  const { jobId } = req.body || {};
  if (!isUuid(jobId)) throw httpError(400, "jobId must be a UUID.");
  await assertJobOwned(jobId, identity.userId);
  return sendJson(res, 200, await recountJob(jobId));
}

async function cancelJob(req: any, res: any, identity: Identity) {
  const { jobId } = req.body || {};
  if (!isUuid(jobId)) throw httpError(400, "jobId must be a UUID.");
  await assertJobOwned(jobId, identity.userId);

  await supabaseRest(`verification_jobs?id=eq.${jobId}`, {
    method: "PATCH",
    headers: { Prefer: "return=minimal" },
    body: JSON.stringify({ status: "cancelled", updated_at: new Date().toISOString() }),
  });
  void logActivity(identity, { route: "batch-job", action: "Cancelled batch job", statusCode: 200 });
  return sendJson(res, 200, await recountJob(jobId));
}

/* ── shared ──────────────────────────────────────────────────────────────── */

interface JobRow {
  id: string;
  label: string | null;
  status: string;
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  created_at: string;
  updated_at: string;
}

interface ItemRow {
  id: string;
  job_id: string;
  document_id: string | null;
  file_name: string;
  status: string;
  trust_score: number | null;
  total_claims: number;
  needs_review_claims: number;
  error_detail: string | null;
  position: number;
}

async function assertJobOwned(jobId: string, userId: string): Promise<void> {
  const rows = await restJson<Array<{ user_id: string }>>(`verification_jobs?id=eq.${jobId}&select=user_id&limit=1`);
  if (!rows[0]) throw httpError(404, "Job not found.");
  if (rows[0].user_id !== userId) throw httpError(403, "Job belongs to another account.");
}

/**
 * Recompute job counters from its items rather than incrementing.
 * A client that retries an item must not double-count, and a refresh mid-run must converge.
 */
async function recountJob(jobId: string) {
  const items = await restJson<ItemRow[]>(
    `verification_job_items?job_id=eq.${jobId}&select=id,job_id,document_id,file_name,status,trust_score,total_claims,needs_review_claims,error_detail,position&order=position.asc`,
  );
  const completed = items.filter((item) => item.status === "completed").length;
  const failed = items.filter((item) => item.status === "failed").length;

  const [job] = await restJson<JobRow[]>(`verification_jobs?id=eq.${jobId}&select=*&limit=1`);
  const settled = completed + failed >= items.length && items.length > 0;
  const nextStatus = job.status === "cancelled" ? "cancelled" : settled ? (failed === items.length ? "failed" : "completed") : "processing";

  const changed = job.completed_documents !== completed || job.failed_documents !== failed || job.status !== nextStatus;
  if (changed) {
    await supabaseRest(`verification_jobs?id=eq.${jobId}`, {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({ completed_documents: completed, failed_documents: failed, status: nextStatus, updated_at: new Date().toISOString() }),
    });
  }

  return {
    job: toJob({ ...job, completed_documents: completed, failed_documents: failed, status: nextStatus }),
    items: items.map(toItem),
  };
}

const toJob = (job: JobRow) => ({
  id: job.id,
  label: job.label ?? "Batch verification",
  status: job.status,
  totalDocuments: job.total_documents,
  completedDocuments: job.completed_documents,
  failedDocuments: job.failed_documents,
  createdAt: job.created_at,
  updatedAt: job.updated_at,
});

const toItem = (item: ItemRow) => ({
  id: item.id,
  documentId: item.document_id,
  fileName: item.file_name,
  status: item.status,
  trustScore: item.trust_score,
  totalClaims: item.total_claims,
  needsReviewClaims: item.needs_review_claims,
  errorDetail: item.error_detail,
  position: item.position,
});

const clamp = (value: number) => Math.max(0, Math.min(100, Math.round(value)));

async function write<T>(table: string, rows: unknown): Promise<T> {
  const response = await supabaseRest(table, { method: "POST", headers: { Prefer: "return=representation" }, body: JSON.stringify(rows) });
  if (!response.ok) throw new Error(`Writing ${table} failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
  return (await response.json()) as T;
}
