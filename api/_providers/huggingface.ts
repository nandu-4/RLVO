import { parseJson } from "../_gemini.js";
import type { CallOptions, ClaimToVerify, DocumentPayload, ExtractedClaim, ProviderClaimVerdict, RawTextBlock, TranscriptionResult, VisionProviderAdapter } from "./types.js";

const API_URL = "https://router.huggingface.co/v1/chat/completions";
// 72B, not 7B: the HF router (router.huggingface.co) returns 400 model_not_supported for the 7B
// variant on accounts where no enabled provider serves it, which made the Qwen fallback fail
// with an error that named only Gemini and OpenRouter. 72B is verified working with a standard
// HF token. Override via HUGGINGFACE_MODEL.
export const DEFAULT_HUGGINGFACE_MODEL = process.env.HUGGINGFACE_MODEL || "Qwen/Qwen2.5-VL-72B-Instruct";
const transcriptionPrompt = `Transcribe visible document text. Return JSON only: {"documentType":"string","pageCount":1,"blocks":[{"page":1,"text":"string","region":"body","legibility":95,"box_2d":[0,0,100,100]}]}. Never infer.`;
const extractionPrompt = `Extract atomic business facts from this document. Return JSON only: {"claims":[{"field":"string","value":"string","category":"string"}]}. Use only visible text.`;
const verificationPrompt = `Verify each claim only from its supplied candidate evidence. Return JSON only: {"claims":[{"field":"string","status":"verified|corrected|unsupported|needs_review","verified":"string","reason":"string","evidenceIds":["id"],"visionAgreement":0}]}. Never invent evidence IDs.`;

async function call(model: string, prompt: string, document: DocumentPayload, options: CallOptions): Promise<string> {
  const key = process.env.HUGGINGFACE_API_KEY;
  if (!key) throw new Error("HUGGINGFACE_API_KEY is not configured");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 45_000);
  try {
    const response = await fetch(API_URL, { method: "POST", signal: controller.signal, headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model, temperature: 0, max_tokens: 2000, response_format: { type: "json_object" }, messages: [{ role: "user", content: [{ type: "text", text: prompt }, { type: "image_url", image_url: { url: `data:${document.mimeType};base64,${document.data}` } }] }] }) });
    const text = await response.text();
    if (!response.ok) throw new Error(`Hugging Face API ${response.status}: ${text.slice(0, 300)}`);
    const data = JSON.parse(text);
    const content = String(data?.choices?.[0]?.message?.content ?? "").trim();
    if (!content) throw new Error("Hugging Face returned no content");
    return content;
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`Hugging Face call timed out after ${options.timeoutMs ?? 45_000}ms`);
    throw error;
  } finally { clearTimeout(timer); }
}

export function createHuggingFaceAdapter(model = DEFAULT_HUGGINGFACE_MODEL): VisionProviderAdapter {
  return { id: "huggingface", label: `Hugging Face · ${model}`, capability: "document-vision", isConfigured: () => Boolean(process.env.HUGGINGFACE_API_KEY),
    async transcribe(document, options = {}): Promise<TranscriptionResult> { const parsed = parseJson<Partial<TranscriptionResult> | RawTextBlock[]>(await call(model, transcriptionPrompt, document, options)); const blocks = Array.isArray(parsed) ? parsed : parsed.blocks ?? []; return { documentType: Array.isArray(parsed) ? "Unknown document" : parsed.documentType ?? "Unknown document", pageCount: Array.isArray(parsed) ? 1 : Number(parsed.pageCount) || 1, blocks }; },
    async extractClaims(document, options = {}): Promise<ExtractedClaim[]> { const parsed = parseJson<{ claims?: ExtractedClaim[] } | ExtractedClaim[]>(await call(model, extractionPrompt, document, options)); return Array.isArray(parsed) ? parsed : parsed.claims ?? []; },
    async verify(document, claims: ClaimToVerify[], options = {}): Promise<ProviderClaimVerdict[]> { const prompt = `${verificationPrompt}\nClaims: ${JSON.stringify(claims)}`; const parsed = parseJson<{ claims?: ProviderClaimVerdict[] } | ProviderClaimVerdict[]>(await call(model, prompt, document, options)); return Array.isArray(parsed) ? parsed : parsed.claims ?? []; },
  };
}
