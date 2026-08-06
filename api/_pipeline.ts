/**
 * The verification pipeline, extracted so `verify-document` and `benchmark` run exactly the same
 * code. A benchmark that measured a different pipeline than production would be worse than no
 * benchmark at all.
 */
import {
  assembleClaim,
  createStageRecorder,
  predictHallucinationRisk,
  type AssembledClaim,
  type StageEvent,
  type UpstreamClaim,
} from "./_truthlens.js";
import { buildDocumentIndex, type DocumentIndex } from "./_documentIndex.js";
import { retrieveEvidence, type RetrievalReport } from "./_retrieval.js";
import { buildClaimGraph, type ClaimRelation } from "./_signals.js";
import type { ClaimToVerify, DocumentPayload, ProviderClaimVerdict, VisionProviderAdapter } from "./_providers/types.js";
import { ocrConfigured, runPaddleOcr, type OcrEngineId } from "./_ocr.js";

export interface PipelineTimings {
  transcribeMs: number;
  verifyMs: number;
}

export interface OcrProvenance {
  engine: OcrEngineId;
  /** Set when PaddleOCR was configured but unusable and model transcription ran instead. */
  degradedReason?: string;
}

export interface PipelineResult {
  documentType: string;
  quality: DocumentIndex["quality"];
  /** Every OCR block the verification was derived from, cited or not. Needed to replay the overlay. */
  textBlocks: DocumentIndex["blocks"];
  claims: AssembledClaim[];
  relations: ClaimRelation[];
  ocr: OcrProvenance;
  summary: {
    totalClaims: number;
    verifiedCount: number;
    correctedCount: number;
    unsupportedCount: number;
    needsReviewCount: number;
    trustScore: number;
    riskLevel: string;
  };
  stages: StageEvent[];
  timings: PipelineTimings;
}

export interface PipelineOptions {
  transcribeTimeoutMs: number;
  verifyTimeoutMs: number;
  /** Reuse an index built by an earlier run — the benchmark shares one transcription. */
  reuseIndex?: DocumentIndex;
  recorder?: ReturnType<typeof createStageRecorder>;
}

export async function runPipeline(
  provider: VisionProviderAdapter,
  document: DocumentPayload,
  claims: UpstreamClaim[],
  options: PipelineOptions,
): Promise<PipelineResult> {
  const stages = options.recorder ?? createStageRecorder();
  const timings: PipelineTimings = { transcribeMs: 0, verifyMs: 0 };

  /* ── Text extraction — deterministic OCR first, model transcription only as a fallback ── */
  let index = options.reuseIndex;
  let ocr: OcrProvenance = { engine: "paddleocr" };

  if (!index) {
    const startedAt = Date.now();
    let transcription;

    if (ocrConfigured()) {
      try {
        const outcome = await runPaddleOcr(document);
        transcription = outcome;
        ocr = { engine: outcome.engine };
      } catch (error) {
        // Degrade rather than fail: the user still gets a verification, and the response records
        // that a non-deterministic engine produced the text.
        const reason = error instanceof Error ? error.message : "OCR service unreachable";
        transcription = await provider.transcribe(document, { timeoutMs: options.transcribeTimeoutMs });
        ocr = { engine: "model-transcription", degradedReason: reason };
      }
    } else {
      transcription = await provider.transcribe(document, { timeoutMs: options.transcribeTimeoutMs });
      ocr = { engine: "model-transcription", degradedReason: "OCR_SERVICE_URL is not configured on this deployment." };
    }

    timings.transcribeMs = Date.now() - startedAt;
    index = buildDocumentIndex(transcription);
    if (index.blocks.length === 0) {
      throw new Error("No readable text could be extracted from this document, so no claim can be verified against it.");
    }
    stages.record(
      "text_extraction",
      ocr.engine === "paddleocr" ? "Text extraction (PaddleOCR)" : "Text extraction (model fallback)",
      `Extracted ${index.blocks.length} text block(s) across ${index.quality.pageCount} page(s) at ${index.quality.meanLegibility}% mean confidence.` +
        (ocr.degradedReason ? ` Deterministic OCR was unavailable — ${ocr.degradedReason}` : " Deterministic: the same page always yields the same text and coordinates.") +
        " This pass never sees the claims.",
      ocr.degradedReason ? "warning" : "info",
    );
  } else {
    ocr = { engine: "model-transcription", degradedReason: "Reused an index built by an earlier run." };
  }

  /* ── Evidence retrieval — independent search, no model involved ── */
  const retrievals = new Map<string, RetrievalReport>();
  for (const claim of claims) {
    retrievals.set(claimKey(claim), retrieveEvidence(index, claim.field, claim.value));
  }
  const totalCandidates = [...retrievals.values()].reduce((sum, report) => sum + report.candidates.length, 0);
  stages.record(
    "evidence_retrieval",
    "Evidence retrieval",
    `Searched ${index.quality.blockCount} indexed block(s) and returned ${totalCandidates} ranked candidate(s) across ${claims.length} claim(s).`,
  );

  /* ── Risk prediction — before verification, so it is actionable ── */
  const risks = new Map<string, ReturnType<typeof predictHallucinationRisk>>();
  for (const claim of claims) {
    const report = retrievals.get(claimKey(claim)) as RetrievalReport;
    risks.set(claimKey(claim), predictHallucinationRisk(claim.field, claim.value, report.candidates, index.quality));
  }
  const highRisk = [...risks.values()].filter((risk) => risk.level === "HIGH").length;
  stages.record(
    "risk_prediction",
    "Hallucination risk prediction",
    `Flagged ${highRisk} claim(s) as high pre-verification risk from retrieval strength and page legibility.`,
    highRisk > 0 ? "warning" : "info",
  );

  /* ── Verification — provider may only cite retrieved candidates ── */
  const toVerify: ClaimToVerify[] = claims.map((claim) => ({
    field: claim.field,
    value: claim.value,
    candidates: (retrievals.get(claimKey(claim)) as RetrievalReport).candidates.map((candidate) => ({
      id: candidate.block.id,
      page: candidate.block.page,
      text: candidate.block.text,
      region: candidate.block.region,
    })),
  }));
  const verifyStartedAt = Date.now();
  const verdicts = await provider.verify(document, toVerify, { timeoutMs: options.verifyTimeoutMs });
  timings.verifyMs = Date.now() - verifyStartedAt;
  const verdictByField = indexVerdicts(verdicts);
  stages.record("verification", "Claim verification", `Provider returned ${verdicts.length} verdict(s), each restricted to citing retrieved evidence.`);

  /* ── Reflection + trust scoring ── */
  const assembled = claims.map((claim, position) =>
    assembleClaim({
      claim,
      index: position,
      retrieval: retrievals.get(claimKey(claim)) as RetrievalReport,
      risk: risks.get(claimKey(claim)) as ReturnType<typeof predictHallucinationRisk>,
      verdict: takeVerdict(verdictByField, claim, verdicts, position),
      quality: index.quality,
    }),
  );

  const count = (status: string) => assembled.filter((claim) => claim.status === status).length;
  const needsReviewCount = count("needs_review");
  const trustScore = Math.round(assembled.reduce((total, claim) => total + claim.trustScore, 0) / assembled.length);
  const fullyMeasured = assembled.filter((claim) => claim.confidenceBreakdown.measuredCount >= 4).length;
  stages.record(
    "trust_scoring",
    "Reflection & trust scoring",
    `Scored ${assembled.length} claim(s); ${fullyMeasured} had at least four independently measured signals; ${needsReviewCount} withheld for human review.`,
    needsReviewCount > 0 ? "warning" : "success",
  );

  return {
    documentType: index.documentType,
    quality: index.quality,
    /*
     * The OCR blocks the whole verification was derived from. Returned so a stored session can
     * reconstruct the page — evidence overlays reference these coordinates, and a snapshot that
     * kept only the cited evidence could not redraw the retrieved-but-unused regions that explain
     * why a claim scored the way it did.
     */
    textBlocks: index.blocks,
    claims: assembled,
    relations: buildClaimGraph(
      assembled.map((claim) => ({ id: claim.id, field: claim.field, value: claim.originalValue, blocks: claim.supportingBlocks })),
    ),
    summary: {
      totalClaims: assembled.length,
      verifiedCount: count("verified"),
      correctedCount: count("corrected"),
      unsupportedCount: count("unsupported"),
      needsReviewCount,
      trustScore,
      riskLevel: trustScore >= 85 && needsReviewCount === 0 ? "LOW" : trustScore >= 60 ? "MEDIUM" : "HIGH RISK",
    },
    ocr,
    stages: stages.events,
    timings,
  };
}

export const claimKey = (claim: UpstreamClaim) => `${claim.field} ${claim.value}`;

/** Group verdicts by normalised field so duplicates can be consumed in order. */
function indexVerdicts(verdicts: ProviderClaimVerdict[]): Map<string, ProviderClaimVerdict[]> {
  const byField = new Map<string, ProviderClaimVerdict[]>();
  for (const verdict of verdicts) {
    if (!verdict || typeof verdict.field !== "string") continue;
    const key = verdict.field.trim().toLowerCase();
    byField.set(key, [...(byField.get(key) || []), verdict]);
  }
  return byField;
}

/**
 * Match a verdict back to the claim we asked about, falling back to positional order.
 * A claim the provider skipped simply has no verdict and is assembled as needs_review — far
 * better than discarding an entire multi-claim verification because one result went missing.
 */
function takeVerdict(
  byField: Map<string, ProviderClaimVerdict[]>,
  claim: UpstreamClaim,
  all: ProviderClaimVerdict[],
  position: number,
): ProviderClaimVerdict | undefined {
  const bucket = byField.get(claim.field.trim().toLowerCase());
  if (bucket && bucket.length > 0) return bucket.shift();
  return all.length === 1 ? all[0] : all[position];
}
