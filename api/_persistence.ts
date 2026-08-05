/**
 * Durable write path for a completed verification.
 *
 * Until this ran, `verify-document` returned throwaway `crypto.randomUUID()` ids and claim
 * ids shaped `claim-1`, so every downstream human-review write failed on an invalid-UUID /
 * foreign-key error that the UI never surfaced. Persisting here is what makes claim ids real,
 * which is what makes review, audit trail, and analytics possible at all.
 */
import { supabaseRest, type Workspace } from "./_workspace.js";
import type { AssembledClaim } from "./_truthlens.js";

interface PersistInput {
  workspace: Workspace;
  /** Set when this document is one item of a batch job. */
  jobId?: string | null;
  fileName: string;
  documentType: string;
  fileSizeKb: number;
  modelUsed: string;
  claims: AssembledClaim[];
  summary: { totalClaims: number; verifiedCount: number; correctedCount: number; unsupportedCount: number; needsReviewCount: number; trustScore: number; riskLevel: string };
  timeline: Array<{ step: string; title: string; detail: string; status: string; timestamp: string }>;
  relations: Array<{ from: string; to: string; kind: string; strength: number }>;
  quality: { meanLegibility: number; blockCount: number; pageCount: number };
  retentionDays: number;
}

export interface PersistedIds {
  documentId: string;
  /** In-memory claim id (claim-1, ...) -> durable UUID. */
  claimIds: Record<string, string>;
}

async function insert<T = void>(table: string, rows: unknown, prefer = "return=representation"): Promise<T> {
  const response = await supabaseRest(table, {
    method: "POST",
    headers: { Prefer: prefer },
    body: JSON.stringify(rows),
  });
  if (!response.ok) throw new Error(`Persisting ${table} failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
  // return=minimal responds 201 with an empty body; parsing it as JSON would throw.
  if (prefer.includes("minimal")) return undefined as T;
  return (await response.json()) as T;
}

export async function persistVerification(input: PersistInput): Promise<PersistedIds> {
  const retentionUntil = new Date(Date.now() + input.retentionDays * 86_400_000).toISOString();

  const [document] = await insert<Array<{ id: string }>>("documents", [{
    document_name: input.fileName,
    document_type: input.documentType,
    file_size_kb: input.fileSizeKb,
    model_used: input.modelUsed,
    trust_score: input.summary.trustScore,
    risk_level: input.summary.riskLevel,
    total_claims: input.summary.totalClaims,
    verified_claims: input.summary.verifiedCount,
    corrected_claims: input.summary.correctedCount,
    unsupported_claims: input.summary.unsupportedCount,
    needs_review_claims: input.summary.needsReviewCount,
    organization_id: input.workspace.id,
    job_id: input.jobId ?? null,
    retention_until: retentionUntil,
    processing_status: "completed",
    mean_legibility: input.quality.meanLegibility,
    indexed_blocks: input.quality.blockCount,
    page_count: input.quality.pageCount,
  }]);

  // PostgREST returns bulk-insert representations in request order, so index alignment is sound.
  const claimRows = await insert<Array<{ id: string }>>("claims", input.claims.map((claim) => ({
    document_id: document.id,
    field_name: claim.field,
    category: claim.category,
    original_value: claim.originalValue,
    verified_value: claim.verifiedValue ?? null,
    status: claim.status,
    trust_score: claim.trustScore,
    reason: claim.reason,
    ocr_agreement: claim.confidenceBreakdown.ocrAgreement,
    vision_agreement: claim.confidenceBreakdown.visionAgreement,
    layout_agreement: claim.confidenceBreakdown.layoutAgreement,
    semantic_agreement: claim.confidenceBreakdown.semanticAgreement,
    evidence_strength: claim.confidenceBreakdown.evidenceStrength,
    signals_measured: claim.confidenceBreakdown.measuredCount,
    hallucination_risk_level: claim.hallucinationRisk.level,
    hallucination_risk_score: claim.hallucinationRisk.score,
    // Stored so a report can be regenerated identically months later, without re-running the model.
    score_rationale: { why: claim.confidenceBreakdown.why, basis: claim.confidenceBreakdown.basis, unmeasured: claim.confidenceBreakdown.unmeasured },
    retrieval_candidates: claim.retrieval.candidateCount,
    retrieval_cited: claim.retrieval.citedCount,
  })));

  const claimIds: Record<string, string> = {};
  input.claims.forEach((claim, index) => {
    if (claimRows[index]) claimIds[claim.id] = claimRows[index].id;
  });

  const evidenceRows = input.claims.flatMap((claim) =>
    claim.evidence.map((item) => ({
      claim_id: claimIds[claim.id],
      evidence_type: item.type,
      source_name: item.source,
      extracted_text: item.text,
      page_number: item.pageNumber,
      bounding_box_json: item.boundingBox ?? null,
      layout_region: item.layoutRegion ?? null,
      confidence: item.confidence,
      retrieved_by: item.retrievedBy,
      cited: item.cited,
    })).filter((row) => Boolean(row.claim_id)),
  );
  if (evidenceRows.length > 0) await insert("claim_evidence", evidenceRows, "return=minimal");

  if (input.timeline.length > 0) {
    await insert("verification_timeline", input.timeline.map((event) => ({
      document_id: document.id,
      step_name: event.step,
      event_title: event.title,
      event_detail: event.detail,
      status: event.status,
      timestamp_formatted: event.timestamp,
    })), "return=minimal");
  }

  // Machine decisions belong in the audit trail too — otherwise the trail only ever shows
  // the claims a human happened to touch, which is not an audit trail.
  const auditRows = input.claims
    .map((claim) => ({
      document_id: document.id,
      claim_id: claimIds[claim.id],
      file_name: input.fileName,
      field_name: claim.field,
      original_value: claim.originalValue,
      final_value: claim.verifiedValue ?? claim.originalValue,
      status: claim.status,
      trust_score: claim.trustScore,
      reviewer_name: "Automated Engine",
      reviewer_notes: claim.reason,
    }))
    .filter((row) => Boolean(row.claim_id));
  if (auditRows.length > 0) await insert("audit_trails", auditRows, "return=minimal");

  // Claims needing a human decision become real, assignable work items.
  const reviewTasks = input.claims
    .filter((claim) => claim.status === "needs_review")
    .map((claim) => ({ document_id: document.id, claim_id: claimIds[claim.id], status: "open" }))
    .filter((row) => Boolean(row.claim_id));
  if (reviewTasks.length > 0) await insert("review_tasks", reviewTasks, "return=minimal");

  const relationRows = input.relations
    .map((relation) => ({
      document_id: document.id,
      from_claim_id: claimIds[relation.from],
      to_claim_id: claimIds[relation.to],
      kind: relation.kind,
      strength: relation.strength,
    }))
    .filter((row) => Boolean(row.from_claim_id) && Boolean(row.to_claim_id));
  if (relationRows.length > 0) await insert("claim_relations", relationRows, "return=minimal");

  // Cheap denormalised counter so the dashboard does not COUNT(*) the documents table per load.
  void supabaseRest(`rpc/increment_document_count`, {
    method: "POST",
    headers: { Prefer: "return=minimal" },
    body: JSON.stringify({ workspace: input.workspace.id }),
  }).catch(() => undefined);

  return { documentId: document.id, claimIds };
}
