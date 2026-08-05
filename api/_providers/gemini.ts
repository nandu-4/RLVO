import { callGemini, parseJson } from "../_gemini.js";
import type {
  CallOptions,
  ClaimToVerify,
  DocumentPayload,
  ExtractedClaim,
  ProviderClaimVerdict,
  RawTextBlock,
  TranscriptionResult,
  VisionProviderAdapter,
} from "./types.js";

/**
 * Google Gemini adapter — currently the only implemented provider.
 *
 * Pass 1 (`transcribe`) never sees a claim. That matters: a model asked to "find evidence for
 * X" will find something resembling X. Transcribing first, then retrieving, then verifying is
 * what stops the evidence from being manufactured to fit the answer.
 */

const TRANSCRIBE_PROMPT = `You are a document transcription engine. Transcribe every visible text block in the supplied document exactly as printed.

Rules:
- Transcribe ONLY what is visibly rendered on the page. Never output PDF metadata, xref tables, object streams, TeX or LaTeX internals, producer strings, or any document container information.
- One entry per visually distinct block (a heading, a label/value pair, a table row, a paragraph, a stamp, a caption).
- Preserve original spelling, casing, punctuation, currency symbols and digits. Do not summarise, translate, correct, or infer.
- box_2d is the block's location as [ymin, xmin, ymax, xmax], each an integer 0-1000 normalised to the page, origin at the TOP-LEFT. A line near the top of the page has a small ymin; a line near the bottom has a large ymin. Never let ymin or ymax exceed 1000.
- region is one of: header, body, table, footer, signature, logo, figure.
- legibility is your honest 0-100 read of how clearly that block could be resolved: low for small type, blur, skew, low contrast, partial occlusion or handwriting; high for crisp printed text.
- Do not omit blocks because they seem unimportant.

Return a JSON OBJECT only — never a bare array. The top level must have documentType, pageCount and blocks:
{"documentType":"string","pageCount":1,"blocks":[{"page":1,"text":"ORACLE CORPORATION","region":"header","legibility":98,"box_2d":[18,115,42,610]},{"page":1,"text":"Invoice Number: INV-2024-8891","region":"body","legibility":97,"box_2d":[124,115,140,395]}]}`;

const VERIFY_PROMPT = `You are TruthLens, an evidence-first verification service.

You are given upstream AI claims about a document, and for each claim a list of candidate evidence blocks that were retrieved from that document by a separate search engine. The page image is attached so you can confirm what the text says in context.

Rules:
- Decide each claim ONLY against the supplied candidates and what is visible in the document.
- Cite the candidates you relied on by their exact id in evidenceIds. Never invent an id, never invent evidence text.
- status: "verified" when a candidate supports the claim as stated; "corrected" when a candidate clearly contradicts it AND gives the correct value; "unsupported" when the document positively shows no such fact; "needs_review" when candidates are absent, ambiguous, partially legible, or you are not confident.
- Only set "verified" when the correction comes verbatim from a cited candidate. If you cannot ground a correction in a candidate, return needs_review instead. Never guess a value.
- visionAgreement (0-100): how strongly the rendered page image supports the claim, judged visually rather than from the text alone. Omit it if you did not assess it. Do not copy another score into it.
- reason: one or two sentences explaining the decision, referring to what the evidence actually says.

Return JSON only:
{"claims":[{"field":"string","status":"verified|corrected|unsupported|needs_review","verified":"string","reason":"string","evidenceIds":["string"],"visionAgreement":0}]}`;

const EXTRACT_PROMPT = `You are a business fact extraction engine. From the supplied document, extract every meaningful business fact as an ATOMIC claim.

Rules:
- ATOMIC means exactly one fact per claim. "Education: BTech, KMIT, 9.1 CGPA" is WRONG — emit three claims: Degree, University, CGPA.
- Extract only facts a reader would act on. Never emit PDF metadata, xref, object streams, TeX or LaTeX internals, producer strings, page furniture, or any document container information.
- field is a short human-readable label for what the fact IS. value is exactly what the document says, verbatim — same spelling, casing, digits, currency symbols. Never normalise, translate, correct, or infer.
- Do not invent a field that the document does not state. If a value is illegible, omit the claim entirely rather than guessing.
- category groups related claims (for example Identity, Contact, Financial, Dates, Parties, Clinical, Education). Derive categories from what this document actually contains — do not force a preset scheme.
- Adapt entirely to the document in front of you. Never assume a document type.

Return JSON only:
{"claims":[{"field":"string","value":"string","category":"string"}]}`;

/**
 * Response schemas.
 *
 * Live runs showed the model returning a bare array in place of the documented envelope for both
 * transcription and verification — the CONTENT was correct each time, but a strict parser threw
 * it away and every claim collapsed to needs_review. Prompt wording alone does not fix envelope
 * drift; constraining the decode does. The array-tolerant parsing below stays as a backstop for
 * providers without structured output.
 */
const REGIONS = ["header", "body", "table", "footer", "signature", "logo", "figure"];

const TRANSCRIBE_SCHEMA = {
  type: "object",
  properties: {
    documentType: { type: "string" },
    pageCount: { type: "integer" },
    blocks: {
      type: "array",
      items: {
        type: "object",
        properties: {
          page: { type: "integer" },
          text: { type: "string" },
          region: { type: "string", enum: REGIONS },
          legibility: { type: "integer" },
          box_2d: { type: "array", items: { type: "integer" } },
        },
        required: ["page", "text", "region", "legibility", "box_2d"],
      },
    },
  },
  required: ["documentType", "pageCount", "blocks"],
} as const;

const VERIFY_SCHEMA = {
  type: "object",
  properties: {
    claims: {
      type: "array",
      items: {
        type: "object",
        properties: {
          field: { type: "string" },
          status: { type: "string", enum: ["verified", "corrected", "unsupported", "needs_review"] },
          verified: { type: "string" },
          reason: { type: "string" },
          evidenceIds: { type: "array", items: { type: "string" } },
          visionAgreement: { type: "integer" },
        },
        required: ["field", "status", "reason", "evidenceIds"],
      },
    },
  },
  required: ["claims"],
} as const;

const EXTRACT_SCHEMA = {
  type: "object",
  properties: {
    claims: {
      type: "array",
      items: {
        type: "object",
        properties: { field: { type: "string" }, value: { type: "string" }, category: { type: "string" } },
        required: ["field", "value"],
      },
    },
  },
  required: ["claims"],
} as const;

/**
 * @param model Overrides the deployment default. The benchmark uses this to run one document
 *              through several Gemini models and compare them like-for-like.
 */
export function createGeminiAdapter(model?: string): VisionProviderAdapter {
  return {
    id: "gemini",
    label: `Gemini · ${model ?? "default"}`,
    capability: "document-vision",

    isConfigured: () => Boolean(process.env.GEMINI_API_KEY),

    async transcribe(document: DocumentPayload, options: CallOptions = {}): Promise<TranscriptionResult> {
      const raw = await callGemini(
        [
          { text: `Document name: ${document.fileName}` },
          { inlineData: { mimeType: document.mimeType, data: document.data } },
        ],
        {
          system: TRANSCRIBE_PROMPT,
          maxTokens: 8000,
          temperature: 0,
          // Transcription is OCR + layout, not reasoning. "auto" cost 13s and 1507 thinking
          // tokens on a one-page invoice and timed out on denser pages.
          thinkingBudget: "minimal",
          timeoutMs: options.timeoutMs,
          model,
          responseSchema: TRANSCRIBE_SCHEMA as unknown as Record<string, unknown>,
        },
      );
      const parsed = parseJson<Partial<TranscriptionResult> | RawTextBlock[]>(raw);
      // Models drift on envelope shape — observed returning a bare block array instead of the
      // documented object. Treating that as a hard failure would throw away a perfectly good
      // transcription over a missing wrapper.
      const blocks = Array.isArray(parsed) ? parsed : Array.isArray(parsed.blocks) ? parsed.blocks : [];
      const envelope = Array.isArray(parsed) ? {} : parsed;
      return {
        documentType: typeof envelope.documentType === "string" ? envelope.documentType : "Unknown document",
        pageCount: Number.isFinite(envelope.pageCount)
          ? Number(envelope.pageCount)
          : Math.max(1, ...blocks.map((block) => (Number.isInteger(block?.page) ? Number(block.page) : 1))),
        blocks,
      };
    },

    async verify(document: DocumentPayload, claims: ClaimToVerify[], options: CallOptions = {}): Promise<ProviderClaimVerdict[]> {
      const payload = claims.map((claim) => ({
        field: claim.field,
        claimedValue: claim.value,
        candidates: claim.candidates.map((candidate) => ({
          id: candidate.id,
          page: candidate.page,
          region: candidate.region,
          text: candidate.text,
        })),
      }));

      const raw = await callGemini(
        [
          { text: `Document name: ${document.fileName}\n\nClaims and their retrieved candidate evidence:\n${JSON.stringify(payload, null, 1)}` },
          { inlineData: { mimeType: document.mimeType, data: document.data } },
        ],
        {
          system: VERIFY_PROMPT,
          maxTokens: 8000,
          temperature: 0.05,
          thinkingBudget: "auto",
          timeoutMs: options.timeoutMs,
          model,
          responseSchema: VERIFY_SCHEMA as unknown as Record<string, unknown>,
        },
      );
      const parsed = parseJson<{ claims?: ProviderClaimVerdict[] } | ProviderClaimVerdict[]>(raw);
      return Array.isArray(parsed) ? parsed : Array.isArray(parsed.claims) ? parsed.claims : [];
    },

    async extractClaims(document: DocumentPayload, options: CallOptions = {}): Promise<ExtractedClaim[]> {
      const raw = await callGemini(
        [
          { text: `Document name: ${document.fileName}` },
          { inlineData: { mimeType: document.mimeType, data: document.data } },
        ],
        {
          system: EXTRACT_PROMPT,
          maxTokens: 4000,
          temperature: 0.1,
          thinkingBudget: "minimal",
          timeoutMs: options.timeoutMs,
          model,
          responseSchema: EXTRACT_SCHEMA as unknown as Record<string, unknown>,
        },
      );
      const parsed = parseJson<{ claims?: ExtractedClaim[] } | ExtractedClaim[]>(raw);
      return Array.isArray(parsed) ? parsed : Array.isArray(parsed.claims) ? parsed.claims : [];
    },
  };
}
