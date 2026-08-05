/**
 * Vision provider abstraction.
 *
 * The application must never depend on one vendor. Everything above this file speaks in these
 * types; adding a provider means implementing this interface in one new file and registering it,
 * with no change to the verification pipeline, the retrieval engine, or the UI.
 *
 * Note the shape of `verify`: the provider is handed candidate evidence that the retrieval engine
 * already found, and must cite it by id. It cannot mint evidence of its own. That is deliberate —
 * when the model both asserts a fact and supplies its own proof, verification is circular.
 */

/** Vendor gateways. Model names are chosen within a provider — see _providers/index.ts. */
export type ProviderId = "gemini" | "openrouter";

export type ClaimStatus = "verified" | "corrected" | "unsupported" | "needs_review";

export interface DocumentPayload {
  fileName: string;
  mimeType: string;
  /** Base64 without the data-URL prefix. */
  data: string;
}

export interface CallOptions {
  timeoutMs?: number;
}

/* ── Pass 1: transcription (claim-blind) ── */

export type RegionKind = "header" | "body" | "table" | "footer" | "signature" | "logo" | "figure";

export interface RawTextBlock {
  page: number;
  text: string;
  /** Gemini's native [ymin, xmin, ymax, xmax] 0-1000 form. Normalised downstream by _geometry. */
  box_2d?: unknown;
  /** Alternative {x,y,width,height} form for adapters that emit it. */
  boundingBox?: unknown;
  region?: string;
  /** Provider's own read of how legible this block was, 0-100. */
  legibility?: number;
}

export interface TranscriptionResult {
  documentType: string;
  pageCount: number;
  blocks: RawTextBlock[];
}

/* ── Pass 2: verification against retrieved candidates ── */

export interface CandidateEvidence {
  id: string;
  page: number;
  text: string;
  region: RegionKind;
}

export interface ClaimToVerify {
  field: string;
  value: string;
  candidates: CandidateEvidence[];
}

export interface ProviderClaimVerdict {
  field: string;
  status: ClaimStatus;
  /** Present only when the provider corrects the claim; must be supported by a cited candidate. */
  verified?: string;
  reason: string;
  /** Ids of the candidates the provider actually relied on. Unknown ids are discarded. */
  evidenceIds: string[];
  /**
   * The one signal only the model can supply: how well the rendered page image supports the
   * claim, beyond what the extracted text says. Omitted when the provider did not assess it.
   */
  visionAgreement?: number;
}

/* ── Optional pass: business fact / atomic claim extraction ── */

export interface ExtractedClaim {
  field: string;
  value: string;
  category?: string;
}

export interface VisionProviderAdapter {
  readonly id: ProviderId;
  readonly label: string;
  readonly capability: "document-vision";
  /** False when this deployment has no credentials or implementation for the provider. */
  isConfigured(): boolean;
  transcribe(document: DocumentPayload, options?: CallOptions): Promise<TranscriptionResult>;
  verify(document: DocumentPayload, claims: ClaimToVerify[], options?: CallOptions): Promise<ProviderClaimVerdict[]>;
  /**
   * Propose atomic business facts found in the document, for self-check mode.
   *
   * These are treated exactly like third-party AI claims: they go through the same independent
   * retrieval and verification. That is weaker evidence than checking another system's output —
   * the proposer and the verifier share a failure mode — and the UI says so.
   */
  extractClaims(document: DocumentPayload, options?: CallOptions): Promise<ExtractedClaim[]>;
}
