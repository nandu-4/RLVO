import { clientSafeError, errorMessage, sendJson } from "./_gemini.js";
import { parseUpstreamClaims } from "./_truthlens.js";
import { resolutionChain } from "./_providers/index.js";
import type { DocumentPayload } from "./_providers/types.js";
import { logActivity, persistenceConfigured, resolveWorkspace, statusOf } from "./_workspace.js";
import { callerKey, rateLimit } from "./_ratelimit.js";
import { resolveMediaType, SUPPORTED_MIME } from "./_media.js";

export const maxDuration = 45;

const EXTRACT_TIMEOUT_MS = 35_000;
const MAX_BASE64_CHARS = 6_000_000;

/**
 * Business fact extraction → atomic claim extraction.
 *
 * This is the self-check path: TruthLens proposes the claims itself instead of receiving them
 * from another AI system. The proposed claims are NOT trusted — they are returned to the client,
 * shown for editing, and then sent through the ordinary verification pipeline where an
 * independent retrieval engine has to find evidence for each one.
 *
 * HONEST CAVEAT, surfaced in the response and repeated in the UI: when the same model family
 * both proposes a fact and later verifies it, the two passes share a failure mode. A fact
 * misread identically twice will verify cleanly. Checking a *different* system's output is
 * strictly stronger evidence, and remains the primary mode.
 */
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });
  const startedAt = Date.now();

  try {
    const { image, fileName = "document", provider: requestedProvider, model: requestedModel } = req.body || {};
    if (typeof image !== "string" || image.length === 0) {
      return sendJson(res, 400, { error: "Missing document image or PDF data." });
    }
    if (image.length > MAX_BASE64_CHARS) {
      return sendJson(res, 413, { error: "Document exceeds the upload limit." });
    }
    const chain = resolutionChain(requestedProvider, requestedModel);
    if (chain.length === 0) {
      return sendJson(res, 501, { error: "No AI provider is configured on this deployment. Set GEMINI_API_KEY or OPENROUTER_API_KEY." });
    }

    const workspace = persistenceConfigured() ? await resolveWorkspace(req).catch(() => null) : null;
    const limit = rateLimit(callerKey(req, workspace?.id), 10, 60_000);
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

    // Same failover policy as verification: try the next configured provider before giving up.
    let proposed: Awaited<ReturnType<typeof chain[0]["adapter"]["extractClaims"]>> | null = null;
    let used = chain[0];
    for (const candidate of chain) {
      try {
        proposed = await candidate.adapter.extractClaims(document, { timeoutMs: EXTRACT_TIMEOUT_MS });
        used = candidate;
        break;
      } catch (error) {
        if (candidate === chain[chain.length - 1]) throw error;
      }
    }
    // Reuse the same sanitiser the verification path uses, so container junk and malformed
    // entries are rejected once, by one rule, rather than twice by two.
    const claims = parseUpstreamClaims(proposed ?? []);
    if (claims.length === 0) {
      return sendJson(res, 422, { error: "No business facts could be extracted from this document. Paste the claims manually instead." });
    }

    void logActivity(workspace?.id ?? null, {
      route: "extract-claims",
      action: `Extracted ${claims.length} candidate claim(s) from ${fileName}`,
      statusCode: 200,
      durationMs: Date.now() - startedAt,
    });

    return sendJson(res, 200, {
      provider: used.providerId,
      providerLabel: used.providerLabel,
      modelUsed: used.model,
      claims: dedupe(claims),
      extractionTimeMs: Date.now() - startedAt,
      caveat:
        "These claims were proposed by the same model family that will verify them, so the two passes share a failure mode — a fact misread identically twice will still verify cleanly. Review and edit them before verifying, and prefer checking another system's output when you have one.",
    });
  } catch (error) {
    return sendJson(res, statusOf(error, 422), { error: clientSafeError(error, "claim extraction").message });
  }
}

/** A repeated field/value pair adds no verification signal and wastes a retrieval pass. */
function dedupe(claims: Array<{ field: string; value: string; category?: string }>) {
  const seen = new Set<string>();
  return claims.filter((claim) => {
    const key = `${claim.field.toLowerCase()}::${claim.value.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
