import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import {
  demoModeReason,
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

/**
 * Cross-document review queue.
 *
 * `review_tasks` rows were being written on every verification and never surfaced anywhere, so
 * "needs review" was a dead end: the spec's chain is Needs Review → Assign → Comment → Decide →
 * Audit → Final decision, and only the last three existed. This endpoint supplies the first two.
 *
 *   POST { action: "list", status? }        -> open work items with their claim context
 *   POST { action: "assign", taskId, assignedName }
 *   POST { action: "unassign", taskId }
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  try {
    const identity = await resolveIdentity(req);
    if (!identity) return sendJson(res, 200, { available: false, reason: demoModeReason(), tasks: [] });

    const limit = rateLimit(callerKey(req, identity.userId), 120, 60_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const { action = "list" } = req.body || {};
    if (action === "assign" || action === "unassign") return assign(req, res, identity, action);
    if (action !== "list") return sendJson(res, 400, { error: "action must be one of: list, assign, unassign." });

    const statusFilter = ["open", "assigned", "resolved"].includes(String(req.body?.status)) ? String(req.body.status) : null;

    const rows = await restJson<TaskRow[]>(
      `review_tasks?select=id,status,assigned_user_id,created_at,resolved_at,` +
        `claims!inner(id,field_name,category,original_value,verified_value,status,trust_score,reason,hallucination_risk_level,retrieval_candidates,retrieval_cited),` +
        `documents!inner(id,document_name,document_type,user_id,created_at)` +
        `&documents.user_id=eq.${identity.userId}` +
        (statusFilter ? `&status=eq.${statusFilter}` : "") +
        `&order=created_at.desc&limit=300`,
    );

    return sendJson(res, 200, {
      available: true,
      counts: {
        open: rows.filter((row) => row.status === "open").length,
        assigned: rows.filter((row) => row.status === "assigned").length,
        resolved: rows.filter((row) => row.status === "resolved").length,
      },
      tasks: rows.map((row) => ({
        id: row.id,
        status: row.status,
        assignedName: row.assigned_user_id === identity.userId ? identity.name : null,
        assignedEmail: row.assigned_user_id === identity.userId ? identity.email : null,
        createdAt: row.created_at,
        resolvedAt: row.resolved_at,
        documentId: row.documents.id,
        documentName: row.documents.document_name,
        documentType: row.documents.document_type,
        claimId: row.claims.id,
        field: row.claims.field_name,
        category: row.claims.category,
        originalValue: row.claims.original_value,
        verifiedValue: row.claims.verified_value,
        claimStatus: row.claims.status,
        trustScore: row.claims.trust_score,
        reason: row.claims.reason,
        // Why this landed in the queue, so a reviewer can triage without opening every item.
        risk: row.claims.hallucination_risk_level,
        evidenceRetrieved: row.claims.retrieval_candidates,
        evidenceCited: row.claims.retrieval_cited,
      })),
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "review queue").message });
  }
}

interface TaskRow {
  id: string;
  status: string;
  assigned_user_id: string | null;
  created_at: string;
  resolved_at: string | null;
  claims: {
    id: string;
    field_name: string;
    category: string | null;
    original_value: string;
    verified_value: string | null;
    status: string;
    trust_score: number;
    reason: string;
    hallucination_risk_level: string | null;
    retrieval_candidates: number;
    retrieval_cited: number;
  };
  documents: { id: string; document_name: string; document_type: string; user_id: string; created_at: string };
}

async function assign(req: any, res: any, identity: { userId: string; name: string; email: string }, action: "assign" | "unassign") {
  const { taskId } = req.body || {};
  if (!isUuid(taskId)) throw httpError(400, "taskId must be a UUID.");

  // Ownership check before the service-role write — a client-supplied id never reaches the
  // database unvalidated.
  const owned = await restJson<Array<{ id: string }>>(
    `review_tasks?id=eq.${taskId}&select=id,documents!inner(user_id)&documents.user_id=eq.${identity.userId}&limit=1`,
  );
  if (!owned[0]) throw httpError(404, "Review task not found in this workspace.");

  // Assignment always means "assign to me": the signed-in reviewer. There is no name to type.
  const response = await supabaseRest(`review_tasks?id=eq.${taskId}&status=neq.resolved`, {
    method: "PATCH",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({
      assigned_user_id: action === "assign" ? identity.userId : null,
      status: action === "assign" ? "assigned" : "open",
    }),
  });
  if (!response.ok) throw new Error(`Assignment failed: ${(await response.text()).slice(0, 200)}`);
  const [updated] = (await response.json()) as Array<{ id: string; status: string; assigned_user_id: string | null }>;
  if (!updated) throw httpError(409, "This task has already been resolved.");

  void logActivity(identity as never, {
    route: "review-queue",
    action: action === "assign" ? `Assigned review task to ${identity.name}` : "Unassigned review task",
    statusCode: 200,
  });

  return sendJson(res, 200, { task: { id: updated.id, status: updated.status, assignedName: updated.assigned_user_id === identity.userId ? identity.name : null } });
}
