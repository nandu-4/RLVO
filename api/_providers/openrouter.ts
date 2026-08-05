import { parseJson } from "../_gemini.js";
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
 * OpenRouter adapter — one gateway, many vendors.
 *
 * Implements exactly the same VisionProviderAdapter contract as the Gemini adapter, so the
 * pipeline, retrieval engine, scoring and UI are untouched. Adding Claude, GPT-4.1 or Qwen VL
 * is now a model-name change, not a code change.
 *
 * Two things differ from the Gemini path and are handled here rather than leaking upward:
 *   - Transport is OpenAI-compatible chat completions, not Google's generateContent.
 *   - Documents attach differently: images as image_url parts, PDFs as a file part with
 *     OpenRouter's file-parser plugin, since most vendors cannot read a raw PDF.
 */

const API_URL = "https://openrouter.ai/api/v1/chat/completions";
const RETRYABLE = new Set([408, 429, 500, 502, 503, 504]);
const MAX_RETRIES = 4; // one extra slot so a 402 down-fit still leaves room to retry transport errors
/**
 * Floor for the 402 down-fit. Measured: a real balance offered 788 tokens and the previous floor
 * of 800 refused to retry over a 12-token margin, failing a request that would have worked.
 * 400 tokens still holds a short verification verdict; below that the reply truncates.
 */
const MIN_USABLE_TOKENS = 400;

/*
 * Output budgets sized to what the prompts actually return, not to the model's ceiling.
 *
 * Requesting 8000 made OpenRouter reserve 8000 tokens' worth of balance up front, so a small
 * account was refused for work that needs a fraction of it — measured replies are ~1200-1400
 * tokens for transcription and well under 1000 for verification. A tight budget makes a modest
 * balance go several times further, and truncation is caught by the tolerant JSON parser anyway.
 */
const TRANSCRIBE_MAX_TOKENS = 4000;
const VERIFY_MAX_TOKENS = 2000;
const EXTRACT_MAX_TOKENS = 1500;

/** Vision-capable models known to work through OpenRouter for document work. */
export const OPENROUTER_MODELS = [
  "anthropic/claude-sonnet-4",
  "openai/gpt-4.1",
  "google/gemini-2.5-flash",
  "qwen/qwen2.5-vl-72b-instruct",
] as const;

export const DEFAULT_OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || OPENROUTER_MODELS[0];

/* ── Prompts: identical intent to the Gemini adapter, phrased for chat completions ── */

const TRANSCRIBE_PROMPT = `You are a document transcription engine. Transcribe every visible text block in the supplied document exactly as printed.

Rules:
- Transcribe ONLY what is visibly rendered. Never output PDF metadata, xref tables, object streams, TeX internals, producer strings, or document container information.
- One entry per visually distinct block (a heading, a label/value pair, a table row, a paragraph, a stamp, a caption).
- Preserve original spelling, casing, punctuation, currency symbols and digits. Do not summarise, translate, correct or infer.
- box_2d is [ymin, xmin, ymax, xmax], each an integer 0-1000 normalised to the page, origin TOP-LEFT. Never exceed 1000. Omit box_2d entirely if you cannot locate the block.
- region is one of: header, body, table, footer, signature, logo, figure.
- legibility is your honest 0-100 read of how clearly the block resolved.
- Do not omit blocks because they seem unimportant.

Respond with a JSON OBJECT only, no prose and no markdown fence:
{"documentType":"string","pageCount":1,"blocks":[{"page":1,"text":"string","region":"header","legibility":98,"box_2d":[18,115,42,610]}]}`;

const VERIFY_PROMPT = `You are TruthLens, an evidence-first verification service.

You receive upstream AI claims about a document, and for each claim a list of candidate evidence blocks retrieved from that document by a separate search engine.

Rules:
- Decide each claim ONLY against the supplied candidates and what is visible in the document.
- Cite the candidates you relied on by their exact id in evidenceIds. Never invent an id or evidence text.
- status: "verified" when a candidate supports the claim as stated; "corrected" when a candidate clearly contradicts it AND supplies the correct value; "unsupported" when the document positively shows no such fact; "needs_review" when candidates are absent, ambiguous, partially legible, or you are not confident.
- A correction must come verbatim from a cited candidate. If you cannot ground it, return needs_review. Never guess.
- visionAgreement (0-100): how strongly the rendered page supports the claim, judged visually. Omit if you did not assess it; never copy another score into it.
- reason: one or two sentences referring to what the evidence actually says.

Respond with a JSON OBJECT only, no prose and no markdown fence:
{"claims":[{"field":"string","status":"verified|corrected|unsupported|needs_review","verified":"string","reason":"string","evidenceIds":["string"],"visionAgreement":0}]}`;

const EXTRACT_PROMPT = `You are a business fact extraction engine. Extract every meaningful business fact from the document as an ATOMIC claim.

Rules:
- ATOMIC means exactly one fact per claim. "Education: BTech, KMIT, 9.1 CGPA" is WRONG — emit three claims.
- Extract only facts a reader would act on. Never emit PDF metadata, xref, object streams, producer strings or page furniture.
- field is a short human-readable label. value is exactly what the document says, verbatim. Never normalise, translate, correct or infer.
- Never invent a field the document does not state. Omit illegible values rather than guessing.
- category groups related claims; derive categories from this document, do not force a preset scheme.
- Adapt entirely to the document in front of you. Never assume a document type.

Respond with a JSON OBJECT only, no prose and no markdown fence:
{"claims":[{"field":"string","value":"string","category":"string"}]}`;

/* ── Transport ── */

interface ContentPart {
  type: string;
  text?: string;
  image_url?: { url: string };
  file?: { filename: string; file_data: string };
}

/**
 * Attach the document in the form the vendor can actually read.
 * PDFs go through OpenRouter's file-parser plugin because most vision models accept images only.
 */
function documentParts(document: DocumentPayload): { parts: ContentPart[]; needsPdfPlugin: boolean } {
  const dataUrl = `data:${document.mimeType};base64,${document.data}`;
  if (document.mimeType === "application/pdf") {
    return { parts: [{ type: "file", file: { filename: document.fileName, file_data: dataUrl } }], needsPdfPlugin: true };
  }
  return { parts: [{ type: "image_url", image_url: { url: dataUrl } }], needsPdfPlugin: false };
}

async function callOpenRouter(
  model: string,
  system: string,
  userText: string,
  document: DocumentPayload,
  options: CallOptions & { maxTokens?: number; temperature?: number },
): Promise<string> {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("OPENROUTER_API_KEY is not configured");

  const { parts, needsPdfPlugin } = documentParts(document);
  const payload: Record<string, unknown> = {
    model,
    messages: [
      { role: "system", content: system },
      { role: "user", content: [{ type: "text", text: userText }, ...parts] },
    ],
    max_tokens: options.maxTokens ?? 8000,
    temperature: options.temperature ?? 0,
    // Ask for JSON where the vendor supports it; the tolerant parser covers the rest.
    response_format: { type: "json_object" },
  };
  if (needsPdfPlugin) {
    payload.plugins = [{ id: "file-parser", pdf: { engine: "pdf-text" } }];
  }

  const timeoutMs = options.timeoutMs ?? 45_000;
  const deadline = Date.now() + timeoutMs;
  let lastError = "";

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error(lastError || `OpenRouter call exceeded ${timeoutMs}ms budget`);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), remaining);
    let response: Response;
    try {
      response = await fetch(API_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${key}`,
          "Content-Type": "application/json",
          // OpenRouter attributes usage with these; harmless if unset.
          "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "https://truthlens.local",
          "X-Title": "TruthLens AI",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      if (controller.signal.aborted) throw new Error(`OpenRouter call timed out after ${timeoutMs}ms`);
      lastError = `OpenRouter request failed: ${err instanceof Error ? err.message : "unknown"}`;
      if (attempt === MAX_RETRIES - 1) break;
      await sleep(Math.min(1000 * 2 ** attempt, Math.max(0, deadline - Date.now())));
      continue;
    }
    clearTimeout(timer);

    if (response.ok) {
      const data = await response.json();
      // OpenRouter surfaces upstream vendor failures inside a 200 body.
      if (data?.error) throw new Error(`OpenRouter upstream error: ${String(data.error.message ?? data.error).slice(0, 300)}`);
      const text = String(data?.choices?.[0]?.message?.content ?? "").trim();
      if (!text) throw new Error(`OpenRouter returned no content (finish_reason: ${data?.choices?.[0]?.finish_reason ?? "unknown"})`);
      return text;
    }

    const bodyText = (await response.text()).slice(0, 600);
    lastError = `OpenRouter API ${response.status}: ${bodyText}`;

    /*
     * A 402 on a low-balance account is not a dead end. OpenRouter states exactly what the caller
     * can afford — "You requested up to 8000 tokens, but can only afford 2663" — so fit the
     * request to the budget and try once more rather than failing outright. Requesting a fixed
     * 8000 made every call impossible on a free balance even though the work fits comfortably.
     */
    const affordable = Number(/can only afford (\d+)/i.exec(bodyText)?.[1] ?? 0);
    if (response.status === 402 && affordable >= MIN_USABLE_TOKENS && Number(payload.max_tokens) > affordable) {
      payload.max_tokens = affordable;
      continue; // does not consume a retry slot conceptually; the deadline still bounds it
    }

    if (!RETRYABLE.has(response.status)) break;
    await sleep(Math.min(1000 * 2 ** attempt, Math.max(0, deadline - Date.now())));
  }
  throw new Error(lastError);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ── Adapter ── */

export function createOpenRouterAdapter(model: string = DEFAULT_OPENROUTER_MODEL): VisionProviderAdapter {
  return {
    id: "openrouter",
    label: `OpenRouter · ${model}`,
    capability: "document-vision",

    isConfigured: () => Boolean(process.env.OPENROUTER_API_KEY),

    async transcribe(document, options = {}): Promise<TranscriptionResult> {
      const raw = await callOpenRouter(model, TRANSCRIBE_PROMPT, `Document name: ${document.fileName}`, document, {
        ...options, maxTokens: TRANSCRIBE_MAX_TOKENS, temperature: 0,
      });
      const parsed = parseJson<Partial<TranscriptionResult> | RawTextBlock[]>(raw);
      // Same envelope tolerance as the Gemini adapter: models drift between object and bare array.
      const blocks = Array.isArray(parsed) ? parsed : Array.isArray(parsed.blocks) ? parsed.blocks : [];
      const envelope = Array.isArray(parsed) ? {} : parsed;
      return {
        documentType: typeof envelope.documentType === "string" ? envelope.documentType : "Unknown document",
        pageCount: Number.isFinite(envelope.pageCount)
          ? Number(envelope.pageCount)
          : Math.max(1, ...blocks.map((b) => (Number.isInteger(b?.page) ? Number(b.page) : 1))),
        blocks,
      };
    },

    async verify(document, claims: ClaimToVerify[], options = {}): Promise<ProviderClaimVerdict[]> {
      const payload = claims.map((claim) => ({
        field: claim.field,
        claimedValue: claim.value,
        candidates: claim.candidates.map((c) => ({ id: c.id, page: c.page, region: c.region, text: c.text })),
      }));
      const raw = await callOpenRouter(
        model,
        VERIFY_PROMPT,
        `Document name: ${document.fileName}\n\nClaims and their retrieved candidate evidence:\n${JSON.stringify(payload, null, 1)}`,
        document,
        { ...options, maxTokens: VERIFY_MAX_TOKENS, temperature: 0.05 },
      );
      const parsed = parseJson<{ claims?: ProviderClaimVerdict[] } | ProviderClaimVerdict[]>(raw);
      return Array.isArray(parsed) ? parsed : Array.isArray(parsed.claims) ? parsed.claims : [];
    },

    async extractClaims(document, options = {}): Promise<ExtractedClaim[]> {
      const raw = await callOpenRouter(model, EXTRACT_PROMPT, `Document name: ${document.fileName}`, document, {
        ...options, maxTokens: EXTRACT_MAX_TOKENS, temperature: 0.1,
      });
      const parsed = parseJson<{ claims?: ExtractedClaim[] } | ExtractedClaim[]>(raw);
      return Array.isArray(parsed) ? parsed : Array.isArray(parsed.claims) ? parsed.claims : [];
    },
  };
}
