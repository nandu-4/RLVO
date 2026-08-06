import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import {
  assertDocumentOwned,
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

export const maxDuration = 15;

type Decision = "approved" | "rejected" | "overridden";
const DECISIONS: Decision[] = ["approved", "rejected", "overridden"];

interface Feedback {
  status: Decision;
  reviewerNotes?: string;
  overrideValue?: string;
  reviewerName?: string;
}

/**
 * Records a human decision on a claim.
 *
 * There are no accounts, so the reviewer is a self-declared display name rather than a verified
 * identity — the audit trail says "who said they made this decision", not "who provably did".
 * That distinction is recorded honestly rather than dressed up as authentication.
 *
 * What IS enforced: the claim must exist and belong to the caller's workspace, checked before any
 * service-role write. Client-supplied ids never reach the database unvalidated.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });
  const startedAt = Date.now();

  try {
    // Reviewer identity comes from the verified session, never from the request body.
    const identity = await resolveIdentity(req);
    if (!identity) return sendJson(res, 401, { error: demoModeReason() });

    const limit = rateLimit(callerKey(req, identity.userId), 60, 60_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const { documentId, claimId, feedback } = req.body || {};
    if (typeof documentId !== "string" || typeof claimId !== "string" || !feedback) {
      return sendJson(res, 400, { error: "documentId, claimId, and feedback are required." });
    }
    const decision = feedback as Feedback;
    if (!DECISIONS.includes(decision.status)) {
      return sendJson(res, 400, { error: "Invalid human review decision." });
    }
    if (decision.status === "overridden" && !decision.overrideValue?.trim()) {
      return sendJson(res, 400, { error: "An override must supply the corrected value." });
    }
    if (!isUuid(claimId)) throw httpError(400, "claimId must be a persisted UUID.");

    await assertDocumentOwned(documentId, identity);

    const claims = await restJson<Array<{ id: string; field_name: string; original_value: string; verified_value: string | null; status: string; trust_score: number }>>(
      `claims?id=eq.${claimId}&document_id=eq.${documentId}&select=id,field_name,original_value,verified_value,status,trust_score&limit=1`,
    );
    const claim = claims[0];
    if (!claim) throw httpError(404, "Claim not found on this document.");

    // The session is the source of truth; a client-supplied name is ignored entirely.
    const reviewerName = identity.name;

    const [record] = await write<Array<{ id: string; created_at: string }>>("review_decisions", {
      document_id: documentId,
      claim_id: claimId,
      decision: decision.status,
      reviewer_user_id: identity.userId,
      reviewer_notes: decision.reviewerNotes?.trim().slice(0, 2000) || null,
      override_value: decision.overrideValue?.trim().slice(0, 1000) || null,
    });

    const finalValue = decision.overrideValue?.trim() || claim.verified_value || claim.original_value;
    const finalStatus = decision.status === "overridden" ? "corrected" : claim.status;

    // A decision that is not in the audit trail is not auditable.
    await write(
      "audit_trails",
      {
        document_id: documentId,
        claim_id: claimId,
        file_name: await documentName(documentId),
        field_name: claim.field_name,
        original_value: claim.original_value,
        final_value: finalValue,
        status: finalStatus,
        trust_score: claim.trust_score,
        reviewer_user_id: identity.userId,
        reviewer_notes: decision.reviewerNotes?.trim().slice(0, 2000) || null,
      },
      "return=minimal",
    );

    if (decision.status === "overridden") {
      await supabaseRest(`claims?id=eq.${claimId}`, {
        method: "PATCH",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify({ verified_value: decision.overrideValue?.trim(), status: "corrected" }),
      });
    }

    await supabaseRest(`review_tasks?claim_id=eq.${claimId}&status=neq.resolved`, {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({
        status: "resolved",
        assigned_user_id: identity.userId,
        resolved_at: new Date().toISOString(),
      }),
    });

    void logActivity(identity, {
      route: "review-claim",
      action: `${reviewerName} ${decision.status} "${claim.field_name}"`,
      statusCode: 201,
      durationMs: Date.now() - startedAt,
    });

    return sendJson(res, 201, {
      decision: {
        id: record.id,
        status: decision.status,
        reviewerName,
        reviewerEmail: identity.email,
        finalValue,
        finalStatus,
        createdAt: record.created_at,
      },
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "review").message });
  }
}

async function documentName(documentId: string): Promise<string> {
  try {
    const rows = await restJson<Array<{ document_name: string }>>(`documents?id=eq.${documentId}&select=document_name&limit=1`);
    return rows[0]?.document_name ?? "Unknown document";
  } catch {
    return "Unknown document";
  }
}

async function write<T = void>(table: string, row: unknown, prefer = "return=representation"): Promise<T> {
  const response = await supabaseRest(table, { method: "POST", headers: { Prefer: prefer }, body: JSON.stringify(row) });
  if (!response.ok) throw new Error(`Writing ${table} failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
  if (prefer.includes("minimal")) return undefined as T;
  return (await response.json()) as T;
}
