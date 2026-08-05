import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import {
  httpError,
  isUuid,
  logActivity,
  persistenceConfigured,
  resolveWorkspace,
  restJson,
  statusOf,
  supabaseRest,
  unavailableReason,
  workspaceToken,
} from "./_workspace.js";
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
    if (!persistenceConfigured()) return sendJson(res, 200, { available: false, reason: unavailableReason(false), tasks: [] });
    const workspace = await resolveWorkspace(req, { create: false });
    if (!workspace) return sendJson(res, 200, { available: false, reason: unavailableReason(Boolean(workspaceToken(req))), tasks: [] });

    const limit = rateLimit(callerKey(req, workspace.id), 120, 60_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const { action = "list" } = req.body || {};
    if (action === "assign" || action === "unassign") return assign(req, res, workspace.id, action);
    if (action !== "list") return sendJson(res, 400, { error: "action must be one of: list, assign, unassign." });

    const statusFilter = ["open", "assigned", "resolved"].includes(String(req.body?.status)) ? String(req.body.status) : null;

    const rows = await restJson<TaskRow[]>(
      `review_tasks?select=id,status,assigned_name,created_at,resolved_at,` +
        `claims!inner(id,field_name,category,original_value,verified_value,status,trust_score,reason,hallucination_risk_level,retrieval_candidates,retrieval_cited),` +
        `documents!inner(id,document_name,document_type,organization_id,created_at)` +
        `&documents.organization_id=eq.${workspace.id}` +
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
        assignedName: row.assigned_name,
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
  assigned_name: string | null;
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
  documents: { id: string; document_name: string; document_type: string; organization_id: string; created_at: string };
}

async function assign(req: any, res: any, workspaceId: string, action: "assign" | "unassign") {
  const { taskId, assignedName } = req.body || {};
  if (!isUuid(taskId)) throw httpError(400, "taskId must be a UUID.");

  // Ownership check before the service-role write — a client-supplied id never reaches the
  // database unvalidated.
  const owned = await restJson<Array<{ id: string }>>(
    `review_tasks?id=eq.${taskId}&select=id,documents!inner(organization_id)&documents.organization_id=eq.${workspaceId}&limit=1`,
  );
  if (!owned[0]) throw httpError(404, "Review task not found in this workspace.");

  const name = action === "assign" ? String(assignedName ?? "").trim().slice(0, 80) : null;
  if (action === "assign" && !name) throw httpError(400, "assignedName is required.");

  const response = await supabaseRest(`review_tasks?id=eq.${taskId}&status=neq.resolved`, {
    method: "PATCH",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({ assigned_name: name, status: action === "assign" ? "assigned" : "open" }),
  });
  if (!response.ok) throw new Error(`Assignment failed: ${(await response.text()).slice(0, 200)}`);
  const [updated] = (await response.json()) as Array<{ id: string; status: string; assigned_name: string | null }>;
  if (!updated) throw httpError(409, "This task has already been resolved.");

  void logActivity(workspaceId, {
    route: "review-queue",
    action: action === "assign" ? `Assigned review task to ${name}` : "Unassigned review task",
    statusCode: 200,
  });

  return sendJson(res, 200, { task: { id: updated.id, status: updated.status, assignedName: updated.assigned_name } });
}
