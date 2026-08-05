import { clientSafeError, activeModel, errorMessage, sendJson } from "./_gemini.js";
import { createStageRecorder, parseUpstreamClaims } from "./_truthlens.js";
import { runPipeline } from "./_pipeline.js";
import { resolutionChain } from "./_providers/index.js";
import type { DocumentPayload } from "./_providers/types.js";
import {
  demoModeReason,
  isUuid,
  logActivity,
  resolveIdentity,
  statusOf,
  type Identity,
} from "./_identity.js";
import { persistVerification, retentionDaysFor } from "./_persistence.js";
import { callerKey, rateLimit } from "./_ratelimit.js";
import { resolveMediaType, SUPPORTED_MIME } from "./_media.js";

/**
 * Vision + reasoning over a full document is the heaviest call in the product, and it makes two
 * provider calls. Without this export it inherited the 10s platform default.
 */
export const maxDuration = 60;

/*
 * Time budget, split by measured cost rather than guessed evenly.
 *
 * Transcription reads every block on the page and is by far the heavier call — measured at 17.1s
 * against 8.7s for verification on a one-page invoice. The old split (26s/24s) gave it barely
 * more time than the cheap step, so a denser document or a slower model tipped it straight into
 * "Gemini call timed out after 26000ms". Budgets are now derived from the function's own
 * maxDuration so the two stay in step if it ever changes.
 */
const REQUEST_BUDGET_MS = 56_000; // maxDuration (60s) less headroom for intake, scoring and the write
const TRANSCRIBE_SHARE = 0.62;
const MIN_VERIFY_MS = 15_000;
/** Least time in which a second provider could plausibly complete both passes. */
const MIN_FAILOVER_MS = 8_000;
/** ~4.5MB of binary once base64 is decoded — matches the platform request body ceiling. */
const MAX_BASE64_CHARS = 6_000_000;
const RATE_LIMIT = Number(process.env.VERIFY_RATE_LIMIT || 10);
const RATE_WINDOW_MS = Number(process.env.VERIFY_RATE_WINDOW_MS || 60_000);

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  const startedAt = Date.now();
  const stages = createStageRecorder(startedAt);

  try {
    /* ── Stage 1: intake ── */
    const { image, fileName = "document", provider: requestedProvider, model: requestedModel, upstreamClaims, jobId } = req.body || {};
    if (typeof image !== "string" || image.length === 0) {
      return sendJson(res, 400, { error: "Missing document image or PDF data." });
    }
    if (image.length > MAX_BASE64_CHARS) {
      return sendJson(res, 413, { error: `Document exceeds the ${Math.round((MAX_BASE64_CHARS * 0.75) / 1_048_576)}MB upload limit. Split the document or reduce its resolution.` });
    }
    const claims = parseUpstreamClaims(upstreamClaims);
    if (claims.length === 0) {
      return sendJson(res, 400, { error: "upstreamClaims is required. Provide the AI-generated claims that TruthLens must verify." });
    }
    // Preferred provider first, then every other configured one — see resolutionChain.
    const chain = resolutionChain(requestedProvider, requestedModel);
    if (chain.length === 0) {
      return sendJson(res, 501, { error: "No AI provider is configured on this deployment. Set GEMINI_API_KEY or OPENROUTER_API_KEY." });
    }

    /* ── Identity: signed in → workspace mode, guest → demo mode ── */
    // A failed identity lookup must never fail the verification. The user still gets their result;
    // the response states plainly that it was not stored.
    const identity: Identity | null = await resolveIdentity(req).catch(() => null);

    const limit = rateLimit(callerKey(req, identity?.userId), RATE_LIMIT, RATE_WINDOW_MS);
    if (!limit.allowed) {
      res.setHeader("Retry-After", String(limit.retryAfterSeconds));
      return sendJson(res, 429, { error: `Rate limit exceeded. Retry in ${limit.retryAfterSeconds}s.` });
    }

    const dataUrl = /^data:([^;]+);base64,(.+)$/s.exec(image);
    const payload = dataUrl?.[2] || image;
    // Trust the bytes, not the filename: a PNG saved as ".jpeg" is labelled image/jpeg by the
    // browser, and vendors reject the mismatch with an undiagnosable "Could not process image".
    const media = resolveMediaType(payload, dataUrl?.[1] ?? "", String(fileName));
    if (!SUPPORTED_MIME.includes(media.mimeType as never)) {
      return sendJson(res, 415, { error: `This file is a ${media.mimeType} document, which no configured provider can read. Upload a PDF, PNG, JPEG or WebP.` });
    }
    const document: DocumentPayload = { fileName, mimeType: media.mimeType, data: payload };
    stages.record(
      "intake",
      "Request intake",
      `Validated ${claims.length} upstream claim(s) and decoded a ${document.mimeType} payload.` +
        (media.corrected ? ` The file was labelled ${media.declared} but its contents are ${media.mimeType}; the detected type was used.` : ""),
      media.corrected ? "warning" : "info",
    );

    /* ── Stages 2-6: the shared pipeline, with automatic provider failover ── */
    let result: Awaited<ReturnType<typeof runPipeline>> | null = null;
    let used = chain[0];
    const attempts: string[] = [];

    for (const candidate of chain) {
      // Recompute each time: a failed attempt has already spent part of the budget.
      const remainingMs = Math.max(12_000, REQUEST_BUDGET_MS - (Date.now() - startedAt));
      const transcribeTimeoutMs = Math.round(remainingMs * TRANSCRIBE_SHARE);
      try {
        result = await runPipeline(candidate.adapter, document, claims, {
          transcribeTimeoutMs,
          verifyTimeoutMs: Math.max(MIN_VERIFY_MS, remainingMs - transcribeTimeoutMs),
          recorder: stages,
        });
        used = candidate;
        break;
      } catch (error) {
        attempts.push(`${candidate.providerId}/${candidate.model}: ${errorMessage(error).slice(0, 120)}`);
        const isLast = candidate === chain[chain.length - 1];
        const timeLeft = REQUEST_BUDGET_MS - (Date.now() - startedAt);
        /*
         * Fail over unless there is genuinely no time for another attempt. The previous guard
         * reserved 15s of a 56s budget, so whenever the first provider was slow — which is
         * precisely when a fallback is most wanted — failover was silently skipped and the user
         * saw the first provider's error instead.
         */
        if (isLast || timeLeft < MIN_FAILOVER_MS) {
          /*
           * Report every provider that was tried, not just the last one. Reporting only the final
           * failure made a two-provider outage look like a single-provider problem: a user told
           * "OpenRouter balance too low" had no way to know Gemini had already been tried and had
           * failed for an entirely different reason.
           */
          if (attempts.length > 1) {
            throw Object.assign(new Error(`All configured providers failed. ${attempts.join(" | ")}`), {
              status: 422,
              allProvidersFailed: true,
            });
          }
          throw error;
        }
        stages.record(
          "provider_failover",
          "Provider failover",
          `${candidate.providerLabel} failed (${errorMessage(error).slice(0, 90)}); retrying with the next configured provider.`,
          "warning",
        );
      }
    }
    if (!result) throw new Error(attempts.join(" | ") || "No provider produced a result.");

    /* ── Stage 7: durable write ── */
    let documentId: string | null = null;
    let persistenceError: string | null = null;
    if (identity) {
      try {
        const persisted = await persistVerification({
          identity,
          jobId: isUuid(jobId) ? jobId : null,
          fileName,
          documentType: result.documentType,
          fileSizeKb: Math.round((document.data.length * 0.75) / 1024),
          modelUsed: `${used.providerId}/${used.model}`,
          claims: result.claims,
          summary: result.summary,
          timeline: stages.events,
          relations: result.relations,
          quality: result.quality,
          retentionDays: await retentionDaysFor(identity),
        });
        documentId = persisted.documentId;
        for (const claim of result.claims) claim.id = persisted.claimIds[claim.id] ?? claim.id;
        stages.record("persistence", "Durable write", `Stored under your account; claims, evidence, relations and audit entries written.`, "success");
      } catch (error) {
        // A failed database write can carry the connection URL or service-role key in its message,
        // and both the timeline and the persistence reason are returned to the browser.
        persistenceError = clientSafeError(error, "persistence").message;
        stages.record("persistence", "Durable write", `Verification succeeded but the result could not be stored. ${persistenceError}`, "danger");
      }
    }

    void logActivity(identity, {
      route: "verify-document",
      action: `Verified ${result.claims.length} claim(s) in ${fileName}`,
      statusCode: 200,
      durationMs: stages.totalMs(),
    });

    return sendJson(res, 200, {
      id: crypto.randomUUID(),
      documentId,
      documentType: result.documentType,
      fileName,
      fileSizeKb: Math.round((document.data.length * 0.75) / 1024),
      // Never hardcoded: the UI shows whatever actually ran, including after a failover.
      provider: used.providerId,
      providerLabel: used.providerLabel,
      modelUsed: used.model,
      failover: attempts.length > 0 ? attempts : undefined,
      summary: result.summary,
      claims: result.claims.map(({ supportingBlocks, ...claim }) => claim),
      relations: result.relations,
      documentQuality: result.quality,
      // Which engine actually read the page, so the UI never implies determinism it did not get.
      ocr: result.ocr,
      // Cross-check = claims came from another AI. Self-check = TruthLens proposed them itself.
      verificationMode: req.body?.selfExtracted === true ? "self-check" : "cross-check",
      timeline: stages.events,
      verificationTimeMs: stages.totalMs(),
      createdAt: new Date().toISOString(),
      persistence: {
        persisted: Boolean(documentId),
        mode: identity ? "workspace" : "demo",
        reason: documentId ? null : (persistenceError ?? demoModeReason()),
      },
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 422), { error: clientSafeError(error, "verification").message, timeline: stages.events });
  }
}
