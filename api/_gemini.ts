// Shared Gemini client for the Vercel serverless functions.
// Files starting with "_" are not exposed as routes.
//
// Calls Google's Generative Language API directly with GEMINI_API_KEY —
// no third-party gateway. Get a free key at https://aistudio.google.com/apikey

/*
 * `gemini-flash-latest` is a moving alias, deliberately. Google retired the entire gemini-2.5-*
 * line for NEW projects — a fresh API key gets 404 "no longer available to new users" on the
 * pinned name, so a hardcoded version silently breaks the app for anyone cloning this repo.
 * Pin a specific version via GEMINI_MODEL when you need reproducibility.
 */
const MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";
const endpointFor = (model: string) => `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

const RETRYABLE = new Set([429, 500, 502, 503, 504]);
const MAX_RETRIES = 3;
const DEFAULT_TIMEOUT_MS = 45_000;
/** Current Gemini models reject thinkingBudget 0 with HTTP 400; 128 is the lowest they accept. */
const MIN_THINKING_BUDGET = 128;

/** The model actually configured for this deployment. Surfaced to clients so the UI never mislabels it. */
export const activeModel = MODEL;

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

export interface GeminiOptions {
  system?: string;
  maxTokens?: number;
  temperature?: number;
  /** Milliseconds before the upstream call is aborted. Keep below the function's maxDuration. */
  timeoutMs?: number;
  /**
   * Thinking budget.
   *
   *   "auto"    — omit thinkingConfig; the model decides. Correct for genuine reasoning work.
   *   "minimal" — floor the budget. Correct for mechanical work like transcription.
   *   number    — explicit budget.
   *
   * MEASURED: "auto" on gemini-flash-latest spent 1507 thinking tokens and 13.0s transcribing a
   * one-page invoice, against 3.9s and zero thinking tokens for identical output on flash-lite.
   * On a denser page that overran the request budget entirely. Transcription is OCR, not
   * reasoning — it must not be allowed to think at length.
   *
   * A budget of 0 is NOT usable: current Gemini models reject it with HTTP 400, so "minimal"
   * clamps to the lowest value they actually accept.
   */
  thinkingBudget?: number | "auto" | "minimal";
  /** Optional responseSchema for structured output (Gemini JSON mode). */
  responseSchema?: Record<string, unknown>;
  /** Override the deployment default — used by the benchmark to compare models on one document. */
  model?: string;
}

export async function callGemini(parts: GeminiPart[], opts: GeminiOptions = {}): Promise<string> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("GEMINI_API_KEY is not configured");

  const generationConfig: Record<string, unknown> = {
    temperature: opts.temperature ?? 0.4,
    maxOutputTokens: opts.maxTokens ?? 400,
  };
  const budget = opts.thinkingBudget ?? "minimal";
  if (budget !== "auto") {
    generationConfig.thinkingConfig = { thinkingBudget: budget === "minimal" ? MIN_THINKING_BUDGET : Math.max(MIN_THINKING_BUDGET, budget) };
  }
  if (opts.responseSchema) {
    generationConfig.responseMimeType = "application/json";
    generationConfig.responseSchema = opts.responseSchema;
  }

  const payload: Record<string, unknown> = { contents: [{ role: "user", parts }], generationConfig };
  if (opts.system) payload.systemInstruction = { parts: [{ text: opts.system }] };

  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const deadline = Date.now() + timeoutMs;
  let lastError = "";

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error(lastError || `Gemini call exceeded ${timeoutMs}ms budget`);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), remaining);
    let resp: Response;
    try {
      resp = await fetch(`${endpointFor(opts.model || MODEL)}?key=${key}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      if (controller.signal.aborted) throw new Error(`Gemini call timed out after ${timeoutMs}ms`);
      lastError = `Gemini request failed: ${errorMessage(err)}`;
      if (attempt === MAX_RETRIES - 1) break;
      await sleep(Math.min(1000 * 2 ** attempt, Math.max(0, deadline - Date.now())));
      continue;
    }
    clearTimeout(timer);

    if (resp.ok) {
      const data = await resp.json();
      const candidate = data?.candidates?.[0];
      const text = (candidate?.content?.parts ?? [])
        .map((p: { text?: string }) => p.text ?? "")
        .join("")
        .trim();
      if (!text) {
        // MAX_TOKENS / SAFETY produce an empty part list; say which so the caller can act.
        throw new Error(`Gemini returned no text (finishReason: ${candidate?.finishReason ?? "unknown"})`);
      }
      return text;
    }

    lastError = `Gemini API ${resp.status}: ${(await resp.text()).slice(0, 500)}`;
    if (!RETRYABLE.has(resp.status)) break;
    await sleep(Math.min(1000 * 2 ** attempt, Math.max(0, deadline - Date.now())));
  }
  throw new Error(lastError);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Parse a "JSON only" model reply. Tolerates markdown fences, leading prose, and trailing
 * commentary by extracting the outermost JSON value before parsing. A bare JSON.parse here
 * turns any stray token into a total request failure.
 */
export function parseJson<T>(raw: string): T {
  const cleaned = raw.replace(/```(?:json)?/gi, "").trim();
  const candidates = [cleaned, sliceOutermost(cleaned, "{", "}"), sliceOutermost(cleaned, "[", "]")];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return JSON.parse(candidate) as T;
    } catch {
      /* try the next extraction strategy */
    }
  }
  throw new Error(`Provider returned unparseable JSON: ${cleaned.slice(0, 300)}`);
}

function sliceOutermost(text: string, open: string, close: string): string | null {
  const start = text.indexOf(open);
  const end = text.lastIndexOf(close);
  return start >= 0 && end > start ? text.slice(start, end + 1) : null;
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

/** Fragments that must never reach a browser: infrastructure detail and credential material. */
const INTERNAL_DETAIL =
  /(supabase\.co|localhost|127\.0\.0\.1|postgres(ql)?:\/\/|\/rest\/v1\/|service_role|apikey|Bearer |eyJ[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{10,}|at\s+\w+\s+\(|\.ts:\d+|node_modules)/i;

/**
 * Converts a thrown error into something safe to send to a client.
 *
 * Errors raised deliberately via `httpError` are written for the user and pass through unchanged.
 * Anything else may embed a database URL, a service-role key, a file path or a stack frame — so it
 * is logged in full server-side and replaced with a generic message plus a reference the user can
 * quote. Leaking infrastructure detail through an error string is a real disclosure route, not a
 * theoretical one.
 */
/**
 * Operational failures the user can actually do something about.
 *
 * A generic "reference ABC123" is right for a genuine internal fault, but wrong for quota and
 * timeouts: those are self-explanatory, extremely common on a free tier, and sending the user to
 * hunt through server logs for them wastes everyone's time. These messages carry no infrastructure
 * detail, so they are safe to surface verbatim.
 */
function actionableMessage(raw: string): string | null {
  // Every provider was tried and every one failed. Naming both beats naming whichever happened
  // to be last, which reads as a single-vendor problem.
  if (/^All configured providers failed\./.test(raw)) {
    /*
     * Raw text is "All configured providers failed. <provider>/<model>: err | <provider>/<model>: err".
     *
     * Diagnose each provider from ITS OWN segment. Matching cause-patterns against the whole
     * string cross-contaminates: an earlier version appended "the model may need a different
     * model id" whenever Hugging Face appeared in the chain, so a plain out-of-credit 402 was
     * reported as a configuration error and sent the operator to change HUGGINGFACE_MODEL —
     * which could never have fixed it.
     */
    const segments = raw.replace(/^All configured providers failed\.\s*/, "").split(" | ");
    const causeFor = (provider: RegExp): string | null => {
      const own = segments.filter((s) => provider.test(s)).join(" ");
      if (!own) return null;
      if (/RESOURCE_EXHAUSTED|exceeded your current quota|free_tier_requests/i.test(own)) return "quota";
      if (/depleted your monthly included credits|requires more credits|requires at least \$|openrouter_credits|insufficient_quota/i.test(own)) return "credits";
      if (/Prompt tokens limit exceeded/i.test(own)) return "prompt-cap";
      if (/model_not_supported|not supported by any provider|no longer available to new users|No endpoints found/i.test(own)) return "model";
      if (/timed out after/i.test(own)) return "timeout";
      return "other";
    };

    const gemini = causeFor(/^gemini\//i);
    const openrouter = causeFor(/^openrouter\//i);
    const hugging = causeFor(/^huggingface\//i);

    const describe = (name: string, cause: string | null): string | null => {
      switch (cause) {
        case null: return null;
        case "quota": return `${name}'s free-tier quota is exhausted`;
        case "credits": return `the ${name} account is out of credit`;
        case "prompt-cap": return `${name} rejected the request as too large for a free account`;
        case "model": return `the ${name} model id is not available to this key`;
        case "timeout": return `${name} timed out`;
        default: return `${name} failed`;
      }
    };

    const failures = [describe("Gemini", gemini), describe("OpenRouter", openrouter), describe("Hugging Face", hugging)].filter(Boolean);
    if (failures.length >= 2) {
      // Lead with the remedy that matches what actually went wrong, not a fixed suggestion.
      const fixes: string[] = [];
      if (gemini === "quota") fixes.push("Gemini's free tier resets daily at midnight US Pacific, and each model has its own daily allowance");
      if (openrouter === "credits" || openrouter === "prompt-cap") fixes.push("OpenRouter needs credits at https://openrouter.ai/settings/credits");
      if (hugging === "credits") fixes.push("Hugging Face needs pre-paid credits or a PRO subscription");
      if (hugging === "model") fixes.push("the Hugging Face model id needs changing — see HUGGINGFACE_MODEL");
      const remedy = fixes.length ? ` ${fixes.join(". ")}.` : " Check each provider's quota or balance in Admin → Models.";
      return `Every AI provider is unavailable: ${failures.join("; ")}.${remedy}`;
    }
    return "Every configured AI provider failed for this request. Check Admin → Models to see which providers are configured, and their quota or balance.";
  }
  if (/RESOURCE_EXHAUSTED|exceeded your current quota|free_tier_requests/i.test(raw)) {
    return "The AI provider's request quota is exhausted. Free-tier keys allow 20 requests per day and each verification uses two, so about 10 documents per day. Enable billing on the provider key, or wait for the daily reset (midnight US Pacific).";
  }
  if (/no longer available to new users/i.test(raw)) {
    return "The configured model is not available to this API key — Google retires older models for newly created projects. Set GEMINI_MODEL to a current model such as 'gemini-flash-latest'.";
  }
  if (/depleted your monthly included credits/i.test(raw)) {
    return "The Hugging Face account has used up its monthly included inference credits. Buy pre-paid credits or subscribe to PRO at https://huggingface.co/settings/billing, or switch provider in Admin → Models. This is a billing limit, not a model-id problem — changing HUGGINGFACE_MODEL will not fix it.";
  }
  if (/model_not_supported|not supported by any provider|No endpoints found/i.test(raw)) {
    return "The configured model id is not served by any inference provider enabled on this account. Pick a different model in Admin → Models, or set HUGGINGFACE_MODEL to a model your account can reach.";
  }
  if (/Prompt tokens limit exceeded/i.test(raw)) {
    return "OpenRouter rejected the request because a free account caps how large a prompt may be, and document images exceed it. Add credits at https://openrouter.ai/settings/credits to lift the cap, or switch to another provider in Admin → Models.";
  }
  if (/requires more credits|requires at least \$|openrouter_credits|insufficient_quota/i.test(raw)) {
    return "The OpenRouter account balance is too low for this request. Add credits at https://openrouter.ai/settings/credits, or switch to another provider in Admin → Models. Note that PDF parsing on OpenRouter needs a paid balance; image uploads do not.";
  }
  if (/timed out after (\d+)ms/i.test(raw)) {
    return "The document took too long to process and the request timed out. This usually means the document has many pages or very dense text. Try a smaller document, or split it into parts.";
  }
  if (/API key is not configured|GEMINI_API_KEY is not configured/i.test(raw)) {
    return "No AI provider key is configured on this deployment. Set GEMINI_API_KEY and restart.";
  }
  if (/API_KEY_INVALID|API key not valid/i.test(raw)) {
    return "The configured AI provider key was rejected as invalid. Check GEMINI_API_KEY.";
  }
  return null;
}

export function clientSafeError(err: unknown, route: string): { message: string; reference: string } {
  const reference = Math.random().toString(36).slice(2, 8).toUpperCase();
  const raw = errorMessage(err);
  const intentional = typeof (err as { status?: unknown })?.status === "number";

  const actionable = actionableMessage(raw);
  if (actionable) {
    console.error(`[${route} ${reference}]`, err); // full detail still reaches the log
    return { message: actionable, reference };
  }

  if (intentional && !INTERNAL_DETAIL.test(raw)) return { message: raw, reference };

  console.error(`[${route} ${reference}]`, err);
  return {
    message: `The ${route} service could not complete this request. Reference ${reference}.`,
    reference,
  };
}
