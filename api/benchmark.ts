import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import { parseUpstreamClaims } from "./_truthlens.js";
import { runPipeline } from "./_pipeline.js";
import { benchmarkTargets } from "./_providers/index.js";
import type { DocumentPayload } from "./_providers/types.js";
import { logActivity, persistenceConfigured, resolveWorkspace, statusOf, supabaseRest } from "./_workspace.js";
import { callerKey, rateLimit } from "./_ratelimit.js";

export const maxDuration = 60;

const MAX_BASE64_CHARS = 6_000_000;
const MAX_MODELS = 4;

/**
 * Model benchmark: run one document and one claim set through several models, compare like for
 * like.
 *
 * Two things make this a real evaluation rather than a demo:
 *   1. Every model runs the SAME pipeline as production (`_pipeline.ts`) — same retrieval, same
 *      scoring, same guardrails. Only the model changes.
 *   2. Each model does its own transcription. Sharing one index would be cheaper and faster, but
 *      it would hide the thing most worth measuring: how well a model reads the page.
 *
 * Models the deployment cannot reach are reported as unavailable rather than silently skipped or
 * substituted with the default.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });
  const startedAt = Date.now();

  try {
    const { image, fileName = "document", upstreamClaims, models } = req.body || {};
    if (typeof image !== "string" || image.length === 0) {
      return sendJson(res, 400, { error: "Missing document image or PDF data." });
    }
    if (image.length > MAX_BASE64_CHARS) {
      return sendJson(res, 413, { error: "Document exceeds the upload limit for a benchmark run." });
    }
    const claims = parseUpstreamClaims(upstreamClaims);
    if (claims.length === 0) {
      return sendJson(res, 400, { error: "upstreamClaims is required — the benchmark measures how each model verifies the same claims." });
    }

    const workspace = persistenceConfigured() ? await resolveWorkspace(req).catch(() => null) : null;

    const limit = rateLimit(callerKey(req, workspace?.id), 3, 300_000);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Benchmark runs are limited. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const available = benchmarkTargets();
    const requested = Array.isArray(models) && models.length > 0 ? models.map(String) : available.filter((t) => t.available).map((t) => t.id);
    const targets = available.filter((target) => requested.includes(target.id)).slice(0, MAX_MODELS);
    if (targets.filter((target) => target.available).length === 0) {
      return sendJson(res, 501, { error: "No benchmark-capable models are configured on this deployment.", targets: available });
    }

    const dataUrl = /^data:([^;]+);base64,(.+)$/s.exec(image);
    const document: DocumentPayload = {
      fileName,
      mimeType: dataUrl?.[1] || (String(fileName).toLowerCase().endsWith(".pdf") ? "application/pdf" : "image/jpeg"),
      data: dataUrl?.[2] || image,
    };

    // Sequential, not parallel: concurrent vision calls hit provider rate limits and distort the
    // latency figure, which is one of the things being measured.
    const runId = crypto.randomUUID();
    const results: Array<Record<string, unknown>> = [];
    const budgetMs = 52_000;

    for (const target of targets) {
      if (!target.available || !target.adapter) {
        results.push({ id: target.id, label: target.label, available: false, error: target.reason ?? "No adapter configured." });
        continue;
      }
      const remaining = budgetMs - (Date.now() - startedAt);
      if (remaining < 12_000) {
        results.push({ id: target.id, label: target.label, available: true, skipped: true, error: "Skipped: the run budget was exhausted before this model started." });
        continue;
      }

      const modelStartedAt = Date.now();
      try {
        const outcome = await runPipeline(target.adapter, document, claims, {
          transcribeTimeoutMs: Math.round(remaining * 0.5),
          verifyTimeoutMs: Math.round(remaining * 0.45),
        });
        const evidenceRetrieved = outcome.claims.reduce((sum, claim) => sum + claim.retrieval.candidateCount, 0);
        const evidenceCited = outcome.claims.reduce((sum, claim) => sum + claim.retrieval.citedCount, 0);

        results.push({
          id: target.id,
          label: target.label,
          available: true,
          documentType: outcome.documentType,
          claimsGenerated: outcome.summary.totalClaims,
          verified: outcome.summary.verifiedCount,
          corrections: outcome.summary.correctedCount,
          unsupported: outcome.summary.unsupportedCount,
          needsReview: outcome.summary.needsReviewCount,
          trustScore: outcome.summary.trustScore,
          // "Hallucinations caught": claims this model did not accept as stated.
          hallucinationsCaught: outcome.summary.correctedCount + outcome.summary.unsupportedCount,
          // Decisiveness, not accuracy — we have no ground truth, so we do not claim accuracy.
          decisionRate: pct(outcome.summary.totalClaims - outcome.summary.needsReviewCount, outcome.summary.totalClaims),
          evidenceCitationRate: pct(evidenceCited, evidenceRetrieved),
          signalsMeasuredAvg: round1(
            outcome.claims.reduce((sum, claim) => sum + claim.confidenceBreakdown.measuredCount, 0) / Math.max(outcome.claims.length, 1),
          ),
          blocksTranscribed: outcome.quality.blockCount,
          meanLegibility: outcome.quality.meanLegibility,
          transcribeMs: outcome.timings.transcribeMs,
          verifyMs: outcome.timings.verifyMs,
          totalMs: Date.now() - modelStartedAt,
          claims: outcome.claims.map((claim) => ({
            field: claim.field,
            status: claim.status,
            trustScore: claim.trustScore,
            verifiedValue: claim.verifiedValue ?? null,
          })),
        });
      } catch (error) {
        // Per-model failures are shown in the results table, so they get the same sanitising.
        results.push({ id: target.id, label: target.label, available: true, error: clientSafeError(error, "benchmark").message, totalMs: Date.now() - modelStartedAt });
      }
    }

    if (workspace) void storeBenchmark(workspace.id, runId, results);
    void logActivity(workspace?.id ?? null, {
      route: "benchmark",
      action: `Benchmarked ${results.length} model(s) on ${fileName}`,
      statusCode: 200,
      durationMs: Date.now() - startedAt,
    });

    return sendJson(res, 200, {
      runId,
      fileName,
      claimCount: claims.length,
      results,
      totalMs: Date.now() - startedAt,
      createdAt: new Date().toISOString(),
      // Said in the payload, not just the UI: without labelled ground truth these are behavioural
      // measurements, and calling them "accuracy" would be a fabrication.
      disclaimer:
        "No labelled ground truth is used. These are behavioural measurements — decisiveness, corrections raised, evidence citation, latency — not accuracy scores.",
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 422), { error: clientSafeError(error, "benchmark").message });
  }
}

const pct = (part: number, whole: number) => (whole === 0 ? 0 : Math.round((part / whole) * 1000) / 10);
const round1 = (value: number) => Math.round(value * 10) / 10;

async function storeBenchmark(workspaceId: string, runId: string, results: Array<Record<string, any>>) {
  const rows = results
    .filter((result) => result.available && !result.error && !result.skipped)
    .map((result) => ({
      organization_id: workspaceId,
      run_id: runId,
      provider_id: result.id,
      model_label: result.label,
      claims_generated: result.claimsGenerated ?? 0,
      verified_claims: result.verified ?? 0,
      corrections: result.corrections ?? 0,
      unsupported_claims: result.unsupported ?? 0,
      needs_review_claims: result.needsReview ?? 0,
      trust_score: result.trustScore ?? null,
      accuracy: result.decisionRate ?? null,
      signals_measured_avg: result.signalsMeasuredAvg ?? null,
      evidence_cited: result.evidenceCitationRate ?? 0,
      verification_time_ms: result.totalMs ?? null,
    }));
  if (rows.length === 0) return;
  try {
    await supabaseRest("model_benchmarks", { method: "POST", headers: { Prefer: "return=minimal" }, body: JSON.stringify(rows) });
  } catch {
    /* benchmark persistence is best-effort; the result is already on its way to the client */
  }
}
