/** Server-side TruthLens domain assembly. No browser secrets, no synthetic evidence. */
import type { BoundingBox } from "./_geometry.js";
import type { DocumentIndex, TextBlock } from "./_documentIndex.js";
import type { RetrievalReport, RetrievedCandidate } from "./_retrieval.js";
import { computeSignals, predictHallucinationRisk, type HallucinationRisk, type TrustSignals } from "./_signals.js";
import type { ClaimStatus, ProviderClaimVerdict } from "./_providers/types.js";

export type { ClaimStatus } from "./_providers/types.js";

export interface UpstreamClaim {
  field: string;
  value: string;
  category?: string;
}

export interface AssembledEvidence {
  id: string;
  type: "ocr" | "vision" | "layout" | "retrieval";
  source: string;
  text: string;
  pageNumber: number;
  boundingBox?: BoundingBox;
  confidence: number;
  layoutRegion: string;
  /** Which retrieval strategies surfaced this item — the audit trail of the search itself. */
  retrievedBy: string[];
  /** True when the verifying provider actually relied on it. */
  cited: boolean;
}

const blockedTerms = /\b(pdftex|tex live|xref|endstream|endobj|document metadata|document container|grounding engine)\b|\b\d+\s+\d+\s+obj\b/i;

export function parseUpstreamClaims(value: unknown): UpstreamClaim[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((claim): claim is Record<string, unknown> => Boolean(claim) && typeof claim === "object")
    .map((claim) => ({
      field: String(claim.field || "").trim(),
      value: String(claim.value ?? claim.originalValue ?? "").trim(),
      category: typeof claim.category === "string" ? claim.category.trim() : undefined,
    }))
    .filter((claim) => claim.field.length > 0 && claim.value.length > 0 && !blockedTerms.test(`${claim.field} ${claim.value}`))
    .slice(0, 60);
}

export interface StageEvent {
  id: string;
  timestamp: string;
  step: string;
  title: string;
  detail: string;
  status: "info" | "warning" | "success" | "danger";
  durationMs: number;
}

/**
 * Records real wall-clock durations for the stages that actually execute.
 *
 * The original timeline was five fixed strings at synthetic offsets while the client animated
 * six more with setTimeout. Every event emitted here is a genuine elapsed interval, and a stage
 * that does not run is simply absent.
 */
export function createStageRecorder(startedAt: number = Date.now()) {
  const events: StageEvent[] = [];
  let cursor = startedAt;

  return {
    events,
    record(step: string, title: string, detail: string, status: StageEvent["status"] = "info"): StageEvent {
      const now = Date.now();
      const event: StageEvent = { id: step, timestamp: new Date(now).toISOString(), step, title, detail, status, durationMs: now - cursor };
      cursor = now;
      events.push(event);
      return event;
    },
    totalMs: () => Date.now() - startedAt,
  };
}

export interface ClaimAssembly {
  claim: UpstreamClaim;
  index: number;
  retrieval: RetrievalReport;
  risk: HallucinationRisk;
  verdict?: ProviderClaimVerdict;
  quality: DocumentIndex["quality"];
}

export interface AssembledClaim {
  id: string;
  field: string;
  category: string;
  originalValue: string;
  verifiedValue?: string;
  status: ClaimStatus;
  trustScore: number;
  reason: string;
  evidence: AssembledEvidence[];
  confidenceBreakdown: ReturnType<typeof toBreakdown>;
  hallucinationRisk: HallucinationRisk;
  retrieval: { searched: string[]; strategies: string[]; candidateCount: number; citedCount: number };
  timeline: StageEvent[];
  reasoning: string[];
  /** Blocks backing this claim, used to derive the claim relation graph. */
  supportingBlocks: TextBlock[];
}

/**
 * Assembles one claim from retrieval output and the provider's verdict.
 *
 * Two guardrails are enforced here rather than trusted to the prompt:
 *  - evidence must resolve to a block the retrieval engine returned, so the provider cannot
 *    invent a citation;
 *  - a verified or corrected status with no resolvable evidence is downgraded to needs_review.
 */
export function assembleClaim(input: ClaimAssembly): AssembledClaim {
  const { claim, index, retrieval, verdict, risk, quality } = input;

  const citedIds = new Set((verdict?.evidenceIds || []).map(String));
  const cited = retrieval.candidates.filter((candidate) => citedIds.has(candidate.block.id));
  // A verdict that cites nothing but had candidates still gets them attached as context, marked
  // uncited, so the reviewer can see what the verifier chose to ignore.
  const evidence = retrieval.candidates.map((candidate, position) => toEvidence(candidate, position, citedIds.has(candidate.block.id)));

  const signals = computeSignals({
    field: claim.field,
    claimedValue: claim.value,
    cited,
    retrieved: retrieval.candidates,
    providerVisionAgreement: verdict?.visionAgreement,
    quality,
  });

  const status = decideStatus(verdict, cited);
  const verifiedValue = resolveVerifiedValue(status, verdict, claim, cited);
  // A correction the provider proposed but could not ground gets dropped by resolveVerifiedValue.
  // Without this flag the claim showed status "corrected", no corrected value, and the provider's
  // original reason — leaving a reviewer with no explanation of where the correction went.
  const correctionDropped = status === "corrected" && Boolean(verdict?.verified?.trim()) && !verifiedValue;

  return {
    id: `claim-${index + 1}`,
    field: claim.field,
    category: claim.category || "Business Fact",
    originalValue: claim.value,
    verifiedValue,
    status,
    trustScore: signals.finalTrustScore,
    reason: buildReason(status, verdict, cited, retrieval, correctionDropped),
    evidence,
    confidenceBreakdown: toBreakdown(signals),
    hallucinationRisk: risk,
    retrieval: {
      searched: retrieval.searched,
      strategies: retrieval.strategiesHit,
      candidateCount: retrieval.candidates.length,
      citedCount: cited.length,
    },
    timeline: [],
    reasoning: [
      `Planning: Predicted ${risk.level} hallucination risk for "${claim.field}" before verification, from document legibility and retrieval strength.`,
      `Evidence Search: Retrieval engine searched ${retrieval.searched.join(", ") || "the transcribed text"} and returned ${retrieval.candidates.length} candidate(s) via ${retrieval.strategiesHit.join(", ") || "no matching strategy"}.`,
      `Reflection: Verifier cited ${cited.length} of ${retrieval.candidates.length} candidate(s); ${signals.measuredCount} of 5 trust signals were independently measured.`,
      `Decision: ${status.replace("_", " ")} at ${signals.finalTrustScore}% trust.`,
    ],
    supportingBlocks: cited.map((candidate) => candidate.block),
  };
}

function toEvidence(candidate: RetrievedCandidate, position: number, cited: boolean): AssembledEvidence {
  const { block } = candidate;
  return {
    id: block.id,
    type: block.boundingBox ? "layout" : "ocr",
    source: `${capitalize(block.region)} · page ${block.page}`,
    text: block.text,
    pageNumber: block.page,
    boundingBox: block.boundingBox,
    confidence: Math.round(candidate.score * 100),
    layoutRegion: block.region,
    retrievedBy: candidate.strategies,
    cited,
  };
}

function decideStatus(verdict: ProviderClaimVerdict | undefined, cited: RetrievedCandidate[]): ClaimStatus {
  if (!verdict || typeof verdict.status !== "string") return "needs_review";
  const status = verdict.status;
  if (!["verified", "corrected", "unsupported", "needs_review"].includes(status)) return "needs_review";
  // An automatic decision without resolvable evidence is not a decision.
  if ((status === "verified" || status === "corrected") && cited.length === 0) return "needs_review";
  return status as ClaimStatus;
}

/**
 * A correction must appear in cited evidence. If the provider proposes a value that no cited
 * block contains, the correction is dropped rather than surfaced as verified truth.
 */
function resolveVerifiedValue(
  status: ClaimStatus,
  verdict: ProviderClaimVerdict | undefined,
  claim: UpstreamClaim,
  cited: RetrievedCandidate[],
): string | undefined {
  if (status === "verified") return claim.value;
  if (status !== "corrected") return undefined;
  const proposed = verdict?.verified?.trim();
  if (!proposed) return undefined;
  const grounded = cited.some((candidate) => candidate.block.text.toLowerCase().includes(proposed.toLowerCase()));
  return grounded ? proposed : undefined;
}

function buildReason(
  status: ClaimStatus,
  verdict: ProviderClaimVerdict | undefined,
  cited: RetrievedCandidate[],
  retrieval: RetrievalReport,
  correctionDropped: boolean,
): string {
  if (status === "needs_review" && cited.length === 0) {
    return retrieval.candidates.length === 0
      ? "The evidence retrieval engine found no region of the document matching this claim, so no automatic decision was made."
      : "Candidate evidence was retrieved but the verifier did not rely on any of it, so the decision was withheld for human review.";
  }
  if (correctionDropped) {
    return `Evidence contradicts the claim, but the proposed correction could not be grounded in any cited text, so it was discarded. Confirm the correct value manually. Provider's reasoning: ${verdict?.reason?.trim() || "not supplied"}`;
  }
  if (status === "corrected" && !verdict?.verified) {
    return "Evidence contradicts the claim, but no corrected value could be grounded in the cited text, so this needs human confirmation.";
  }
  return verdict?.reason?.trim() || "Decision derived from retrieved document evidence.";
}

function toBreakdown(signals: TrustSignals) {
  return {
    ocrAgreement: signals.ocrAgreement.value,
    visionAgreement: signals.visionAgreement.value,
    layoutAgreement: signals.layoutAgreement.value,
    semanticAgreement: signals.semanticAgreement.value,
    evidenceStrength: signals.evidenceStrength.value,
    finalTrustScore: signals.finalTrustScore,
    measuredCount: signals.measuredCount,
    basis: {
      ocrAgreement: signals.ocrAgreement.basis,
      visionAgreement: signals.visionAgreement.basis,
      layoutAgreement: signals.layoutAgreement.basis,
      semanticAgreement: signals.semanticAgreement.basis,
      evidenceStrength: signals.evidenceStrength.basis,
    },
    unmeasured: (Object.keys(signals) as Array<keyof TrustSignals>).filter(
      (key) => typeof signals[key] === "object" && !(signals[key] as { measured: boolean }).measured,
    ),
    why: signals.why,
  };
}

const capitalize = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

export { predictHallucinationRisk };
