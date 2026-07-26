// Shared Gemini client for the Vercel serverless functions.
// Files starting with "_" are not exposed as routes.
//
// Calls Google's Generative Language API directly with GEMINI_API_KEY —
// no third-party gateway. Get a free key at https://aistudio.google.com/apikey

const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const API_BASE = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

const RETRYABLE = new Set([429, 500, 502, 503, 504]);
const MAX_RETRIES = 3;

export interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
}

/** Split a data URL (or plain base64) into { mimeType, data }. */
export function splitDataUrl(dataUrl: string): { mimeType: string; data: string } {
  const m = /^data:([^;]+);base64,(.+)$/s.exec(dataUrl);
  if (m) return { mimeType: m[1], data: m[2] };
  return { mimeType: "image/jpeg", data: dataUrl };
}

export function imagePart(dataUrl: string): GeminiPart {
  const { mimeType, data } = splitDataUrl(dataUrl);
  return { inlineData: { mimeType, data } };
}

export async function callGemini(
  parts: GeminiPart[],
  opts: { system?: string; maxTokens?: number; temperature?: number } = {},
): Promise<string> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("GEMINI_API_KEY is not configured");

  const payload: Record<string, unknown> = {
    contents: [{ role: "user", parts }],
    generationConfig: {
      temperature: opts.temperature ?? 0.4,
      maxOutputTokens: opts.maxTokens ?? 400,
      thinkingConfig: { thinkingBudget: 0 },
    },
  };
  if (opts.system) payload.systemInstruction = { parts: [{ text: opts.system }] };

  let lastError = "";
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const resp = await fetch(`${API_BASE}?key=${key}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (resp.ok) {
      const data = await resp.json();
      const outParts = data?.candidates?.[0]?.content?.parts ?? [];
      const text = outParts.map((p: { text?: string }) => p.text ?? "").join("").trim();
      if (!text) throw new Error("Gemini returned empty text");
      return text;
    }

    lastError = `Gemini API ${resp.status}: ${await resp.text()}`;
    if (!RETRYABLE.has(resp.status)) break;
    await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
  }
  throw new Error(lastError);
}

/** Parse a "JSON only" model reply, tolerating markdown fences. */
export function parseJson<T>(raw: string): T {
  return JSON.parse(raw.replace(/```json|```/g, "").trim()) as T;
}

/** Minimal handler helpers (Node runtime, no framework). */
export function sendJson(res: any, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : "Unknown error";
}
