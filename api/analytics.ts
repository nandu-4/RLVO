import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import { demoModeReason, resolveIdentity, restJson, statusOf } from "./_identity.js";

export const maxDuration = 20;

const WINDOW_DAYS = 30;
const MAX_ROWS = 2000;

interface DocumentRow {
  id: string;
  document_type: string;
  trust_score: number;
  risk_level: string;
  total_claims: number;
  verified_claims: number;
  corrected_claims: number;
  unsupported_claims: number;
  needs_review_claims: number;
  provider: string;
  model: string;
  mean_legibility: number | null;
  created_at: string;
}

interface ClaimRow {
  field_name: string;
  status: string;
  trust_score: number;
  signals_measured: number;
  hallucination_risk_level: string | null;
  retrieval_candidates: number;
  retrieval_cited: number;
}

/**
 * Workspace analytics, computed from stored verifications.
 *
 * Everything here is derived from rows this workspace actually produced. There is no sample data
 * and no synthesised trend: an empty workspace returns zeroes and an explicit `hasData: false`,
 * and the dashboard renders that as an empty state rather than inventing a chart.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "GET" && req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  try {
    // The dashboard is a signed-in feature: history requires an account to belong to.
    const identity = await resolveIdentity(req);
    if (!identity) {
      return sendJson(res, 200, { available: false, reason: demoModeReason(), hasData: false });
    }

    const since = new Date(Date.now() - WINDOW_DAYS * 86_400_000).toISOString();
    const [documents, claims, decisions] = await Promise.all([
      restJson<DocumentRow[]>(
        `documents?user_id=eq.${identity.userId}&created_at=gte.${since}` +
          `&select=id,document_type,trust_score,risk_level,total_claims,verified_claims,corrected_claims,unsupported_claims,needs_review_claims,provider,model,mean_legibility,created_at` +
          `&order=created_at.desc&limit=${MAX_ROWS}`,
      ),
      restJson<ClaimRow[]>(
        `claims?select=field_name,status,trust_score,signals_measured,hallucination_risk_level,retrieval_candidates,retrieval_cited,documents!inner(user_id,created_at)` +
          `&documents.user_id=eq.${identity.userId}&documents.created_at=gte.${since}&limit=${MAX_ROWS * 5}`,
      ),
      restJson<Array<{ decision: string; created_at: string }>>(
        `review_decisions?select=decision,created_at,documents!inner(user_id)&documents.user_id=eq.${identity.userId}&limit=${MAX_ROWS}`,
      ),
    ]);

    if (documents.length === 0) {
      return sendJson(res, 200, { available: true, hasData: false, account: { name: identity.name, email: identity.email } });
    }

    const totalClaims = claims.length || documents.reduce((sum, doc) => sum + doc.total_claims, 0);
    const count = (status: string) => claims.filter((claim) => claim.status === status).length;
    const corrected = count("corrected");
    const unsupported = count("unsupported");
    const needsReview = count("needs_review");
    const verified = count("verified");

    return sendJson(res, 200, {
      available: true,
      hasData: true,
      windowDays: WINDOW_DAYS,
      account: { name: identity.name, email: identity.email },
      totals: {
        documents: documents.length,
        claims: totalClaims,
        verified,
        corrected,
        unsupported,
        needsReview,
        reviewDecisions: decisions.length,
      },
      rates: {
        averageTrustScore: mean(documents.map((doc) => doc.trust_score)),
        // Share of claims the engine changed — the headline "how often was the AI wrong" number.
        correctionRate: ratio(corrected, totalClaims),
        unsupportedRate: ratio(unsupported, totalClaims),
        needsReviewRate: ratio(needsReview, totalClaims),
        // A run is a success when every claim reached an automatic decision.
        verificationSuccessRate: ratio(documents.filter((doc) => doc.needs_review_claims === 0).length, documents.length),
        averageSignalsMeasured: claims.length > 0 ? round1(mean(claims.map((claim) => claim.signals_measured))) : 0,
        evidenceCitationRate: ratio(
          claims.reduce((sum, claim) => sum + claim.retrieval_cited, 0),
          claims.reduce((sum, claim) => sum + claim.retrieval_candidates, 0),
        ),
        averageDocumentLegibility: mean(documents.map((doc) => doc.mean_legibility ?? 0).filter((value) => value > 0)),
      },
      // "Most hallucinated fields": ranked by how often a field failed to verify cleanly.
      mostHallucinatedFields: rank(
        claims.filter((claim) => claim.status !== "verified"),
        (claim) => claim.field_name,
        8,
      ),
      topErrorCategories: [
        { label: "Corrected — evidence contradicted the claim", count: corrected },
        { label: "Unsupported — document showed no such fact", count: unsupported },
        { label: "Needs review — evidence absent or ambiguous", count: needsReview },
      ].filter((entry) => entry.count > 0),
      topDocumentTypes: rank(documents, (doc) => doc.document_type || "Unknown document", 6),
      modelComparison: aggregateBy(documents, (doc) => `${doc.provider || "unknown"}/${doc.model || "unknown"}`).map((group) => ({
        label: group.key,
        documents: group.rows.length,
        averageTrustScore: mean(group.rows.map((doc) => doc.trust_score)),
        needsReviewRate: ratio(
          group.rows.reduce((sum, doc) => sum + doc.needs_review_claims, 0),
          group.rows.reduce((sum, doc) => sum + doc.total_claims, 0),
        ),
      })),
      riskDistribution: {
        low: claims.filter((claim) => claim.hallucination_risk_level === "LOW").length,
        medium: claims.filter((claim) => claim.hallucination_risk_level === "MEDIUM").length,
        high: claims.filter((claim) => claim.hallucination_risk_level === "HIGH").length,
      },
      trend: buildTrend(documents),
      recentActivity: documents.slice(0, 8).map((doc) => ({
        documentId: doc.id,
        documentType: doc.document_type,
        trustScore: doc.trust_score,
        riskLevel: doc.risk_level,
        totalClaims: doc.total_claims,
        needsReview: doc.needs_review_claims,
        createdAt: doc.created_at,
      })),
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "analytics").message });
  }
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

const mean = (values: number[]) => (values.length === 0 ? 0 : Math.round(values.reduce((sum, value) => sum + value, 0) / values.length));
const ratio = (part: number, whole: number) => (whole === 0 ? 0 : Math.round((part / whole) * 1000) / 10);
const round1 = (value: number) => Math.round(value * 10) / 10;

function rank<T>(rows: T[], key: (row: T) => string, limit: number) {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const value = key(row) || "Unknown";
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => ({ label, count }));
}

function aggregateBy<T>(rows: T[], key: (row: T) => string) {
  const groups = new Map<string, T[]>();
  for (const row of rows) {
    const value = key(row);
    groups.set(value, [...(groups.get(value) || []), row]);
  }
  return [...groups.entries()].map(([key, rows]) => ({ key, rows }));
}

/** Daily buckets across the window; days with no runs are present with zeroes so the line is honest. */
function buildTrend(documents: DocumentRow[]) {
  const buckets = new Map<string, DocumentRow[]>();
  for (const doc of documents) {
    const day = doc.created_at.slice(0, 10);
    buckets.set(day, [...(buckets.get(day) || []), doc]);
  }
  const days: Array<{ date: string; documents: number; averageTrustScore: number | null; needsReview: number }> = [];
  for (let offset = WINDOW_DAYS - 1; offset >= 0; offset--) {
    const date = new Date(Date.now() - offset * 86_400_000).toISOString().slice(0, 10);
    const rows = buckets.get(date) || [];
    days.push({
      date,
      documents: rows.length,
      averageTrustScore: rows.length > 0 ? mean(rows.map((doc) => doc.trust_score)) : null,
      needsReview: rows.reduce((sum, doc) => sum + doc.needs_review_claims, 0),
    });
  }
  return days;
}
