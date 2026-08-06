/* ── Enterprise AI Hallucination Verification Platform (v2.0 Types) ── */

export type ClaimStatus = "verified" | "corrected" | "unsupported" | "needs_review";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH RISK";

/** Vendor gateways. Model names are chosen within a provider and come from the backend. */
export type VisionProviderId = "gemini" | "openrouter";

export interface ProviderOption {
  id: VisionProviderId;
  label: string;
  vendor: string;
  keyVar: string;
  configured: boolean;
  models: string[];
  defaultModel: string;
}

export interface BoundingBox {
  x: number;      // percentage or pixel offset (0-100%)
  y: number;
  width: number;
  height: number;
}

export type EvidenceType = "ocr" | "vision" | "layout" | "retrieval" | "metadata";

export interface Evidence {
  id: string;
  type: EvidenceType;
  source: string;
  text: string;
  pageNumber: number;
  /** Always percentages of the page, top-left origin — normalised server-side. */
  boundingBox?: BoundingBox;
  confidence: number; // 0-100 retrieval score
  layoutRegion?: string;
  /** Which retrieval strategies surfaced this item. */
  retrievedBy: string[];
  /** True when the verifying provider actually relied on it, false when it was merely retrieved. */
  cited: boolean;
}

/**
 * Stages that genuinely execute on the server. The previous list described a ten-stage
 * pipeline (ocr, vision, claim_extraction, reflection...) that no code performed — the
 * timeline was fixed strings at synthetic offsets. Only add a step here once something
 * measurable runs for it.
 */
export type TimelineStep =
  | "intake"
  | "document_understanding"
  | "evidence_retrieval"
  | "risk_prediction"
  | "verification"
  | "trust_scoring"
  | "persistence";

export interface VerificationTimelineEvent {
  id: string;
  timestamp: string; // ISO-8601, emitted server-side
  step: TimelineStep | string;
  title: string;
  detail: string;
  status: "info" | "warning" | "success" | "danger";
  /** Real elapsed milliseconds for this stage. */
  durationMs: number;
}

export interface HumanFeedback {
  status: "approved" | "rejected" | "overridden";
  reviewerNotes?: string;
  overrideValue?: string;
  timestamp: string;
  reviewerId?: string;
}

export type SignalKey = "ocrAgreement" | "visionAgreement" | "layoutAgreement" | "semanticAgreement" | "evidenceStrength";

export interface ConfidenceBreakdown {
  ocrAgreement: number;       // char-level similarity of claim vs transcribed text, legibility-weighted
  visionAgreement: number;    // provider's visual read of the page — the only model-supplied signal
  layoutAgreement: number;    // coordinate grounding, region fit, spatial coherence
  semanticAgreement: number;  // content-token overlap between claim and cited evidence
  evidenceStrength: number;   // retrieval ranking and legibility of what the engine found
  finalTrustScore: number;    // weighted mean over MEASURED signals only
  /** How many of the five signals were independently measured (0-5). */
  measuredCount: number;
  /** One line explaining what each number was computed from. */
  basis: Record<SignalKey, string>;
  /** Signals excluded from the score because nothing measured them. */
  unmeasured: SignalKey[];
  /** Plain-language reasons the final score is what it is. */
  why: string[];
}

/**
 * Predicted BEFORE verification runs, from document legibility and retrieval strength only —
 * never from the verification outcome. That is what makes it actionable rather than a
 * restatement of the trust score.
 */
export interface HallucinationRisk {
  level: "LOW" | "MEDIUM" | "HIGH";
  score: number;
  reasons: string[];
}

/** What the Evidence Retrieval Engine searched and found for one claim. */
export interface RetrievalTrace {
  searched: string[];
  strategies: string[];
  candidateCount: number;
  citedCount: number;
}

export interface ClaimRelation {
  from: string;
  to: string;
  kind: "same-region" | "same-page" | "shared-evidence" | "lexical";
  strength: number;
}

export interface DocumentQuality {
  blockCount: number;
  pageCount: number;
  meanLegibility: number;
  lowLegibilityRatio: number;
  boundingBoxCoverage: number;
  smallTypeRatio: number;
}

export interface PersistenceState {
  /** False when the result exists only in this browser session. */
  persisted: boolean;
  mode: "workspace" | "demo";
  reason: string | null;
}

export interface Claim {
  id: string;
  field: string;
  category?: string;
  originalValue: string;
  verifiedValue?: string;
  status: ClaimStatus;
  trustScore: number; // 0-100%
  reason: string;     // XAI explainability narrative
  evidence: Evidence[];
  confidenceBreakdown: ConfidenceBreakdown;
  hallucinationRisk: HallucinationRisk;
  retrieval: RetrievalTrace;
  timeline: VerificationTimelineEvent[];
  reasoning?: string[]; // Per-claim agentic reasoning trace: ["Planning", "Evidence Search", "Reflection", "Decision"]
  feedback?: HumanFeedback;
}

export interface VerificationSummary {
  totalClaims: number;
  verifiedCount: number;
  correctedCount: number;
  unsupportedCount: number;
  needsReviewCount: number;
  trustScore: number; // 0-100%
  riskLevel: RiskLevel;
}

export interface VerificationResult {
  id: string;
  /** Null when the run was not persisted; human review requires a durable id. */
  documentId: string | null;
  documentType: string; // e.g. "Resume", "Contract", "Medical Report", "Architecture Diagram", etc.
  fileName: string;
  fileSizeKb: number;
  /** Reported by the server — never hardcoded, and reflects any failover that occurred. */
  provider: VisionProviderId | string;
  providerLabel: string;
  modelUsed: string;
  /** Present when the preferred provider failed and another was used. */
  failover?: string[];
  fallbackUsed?: boolean;
  fallbackReason?: string;
  attempts?: string[];
  replayMode?: boolean;
  /** Which engine read the page: deterministic OCR, or the model fallback. */
  ocr?: { engine: "paddleocr" | "model-transcription"; degradedReason?: string };
  /** cross-check = claims came from another AI system. self-check = TruthLens proposed them. */
  verificationMode?: "cross-check" | "self-check";
  summary: VerificationSummary;
  claims: Claim[];
  /** Relationships derived from where each claim's evidence sits, not from a document schema. */
  relations: ClaimRelation[];
  documentQuality: DocumentQuality;
  timeline: VerificationTimelineEvent[];
  verificationTimeMs: number;
  createdAt: string;
  persistence: PersistenceState;
}

export interface AuditTrailEntry {
  id: string;
  documentId: string;
  fileName: string;
  documentType: string;
  claimId: string;
  field: string;
  originalValue: string;
  finalValue: string;
  status: ClaimStatus;
  trustScore: number;
  reason: string;
  reviewer: string;
  timestamp: string;
}
