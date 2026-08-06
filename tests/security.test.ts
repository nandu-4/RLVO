import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { clientSafeError } from "../api/_gemini.js";
import { httpError } from "../api/_identity.js";

/**
 * Error responses are a real disclosure route: the API talks to a database and a model provider,
 * and both put URLs, keys and file paths into their error strings. Anything returned to a browser
 * has to be scrubbed.
 */
describe("clientSafeError", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it("passes through deliberate, user-facing messages unchanged", () => {
    expect(clientSafeError(httpError(400, "claimId must be a persisted UUID."), "review").message)
      .toBe("claimId must be a persisted UUID.");
  });

  it("never leaks a database URL", () => {
    const out = clientSafeError(new Error("connect ECONNREFUSED postgresql://user:pw@db.abcdef.supabase.co:5432"), "analytics");
    expect(out.message).not.toMatch(/supabase\.co|postgresql:\/\/|pw@/);
    expect(out.message).toContain(out.reference);
  });

  it("never leaks a service-role key or bearer token", () => {
    const out = clientSafeError(new Error("401 apikey eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 rejected"), "workspace");
    expect(out.message).not.toMatch(/eyJ|apikey/);
  });

  it("never leaks a provider API key", () => {
    const out = clientSafeError(new Error("Gemini API 400: key AIzaSyC7xxxxxxxxxxxxxxxxxxxxxxx invalid"), "verification");
    expect(out.message).not.toMatch(/AIza/);
  });

  it("never leaks a stack frame or source path", () => {
    const err = new Error("boom");
    err.message = "boom at persistVerification (/var/task/api/_persistence.ts:42:11)";
    const out = clientSafeError(err, "verification");
    expect(out.message).not.toMatch(/\.ts:\d+|\/var\/task|at \w+ \(/);
  });

  it("scrubs an intentional error that still embeds internal detail", () => {
    // status is set, but the message picked up infrastructure detail — scrub it anyway.
    const out = clientSafeError(httpError(500, "Writing audit_trails failed: https://xyz.supabase.co/rest/v1/"), "review");
    expect(out.message).not.toMatch(/supabase\.co|rest\/v1/);
  });

  it("always yields a quotable reference and logs the detail server-side", () => {
    const out = clientSafeError(new Error("internal https://xyz.supabase.co detail"), "analytics");
    expect(out.reference).toMatch(/^[A-Z0-9]{6}$/);
    expect(console.error).toHaveBeenCalled();
  });

  it("handles non-Error throwables without crashing", () => {
    for (const thrown of [null, undefined, "a string", 42, { nope: true }]) {
      expect(() => clientSafeError(thrown, "verification")).not.toThrow();
    }
  });
});

/**
 * Quota, retired models and timeouts are the three failures a real user actually hits. A generic
 * reference code for these wastes their time — but the explanation must still carry no
 * infrastructure detail.
 */
describe("actionable operational errors", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it("explains an exhausted quota instead of hiding it behind a reference", () => {
    const out = clientSafeError(new Error('Gemini API 429: {"status":"RESOURCE_EXHAUSTED","message":"You exceeded your current quota"}'), "verification");
    expect(out.message).toMatch(/quota is exhausted/i);
    expect(out.message).toMatch(/enable billing|daily reset/i);
  });

  it("explains a retired model and names the fix", () => {
    const out = clientSafeError(new Error("404 This model models/gemini-2.5-flash is no longer available to new users."), "verification");
    expect(out.message).toMatch(/GEMINI_MODEL/);
  });

  it("explains a timeout in terms of the document, not milliseconds", () => {
    const out = clientSafeError(new Error("Gemini call timed out after 26000ms"), "verification");
    expect(out.message).toMatch(/too long to process|many pages/i);
  });

  it("keeps actionable messages free of infrastructure detail", () => {
    for (const raw of [
      'Gemini API 429 at https://xyz.supabase.co: RESOURCE_EXHAUSTED key AIzaSyC7xxxxxxxxxxxx',
      "Gemini call timed out after 26000ms at callGemini (/var/task/api/_gemini.ts:96:44)",
    ]) {
      const out = clientSafeError(new Error(raw), "verification");
      expect(out.message).not.toMatch(/AIza|supabase\.co|\.ts:\d+|\/var\/task/);
    }
  });
});

/**
 * When every provider fails, naming only the last one reads as a single-vendor problem and sends
 * the user to fix the wrong thing.
 */
describe("multi-provider failure", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it("names both causes when quota and balance both fail", () => {
    const out = clientSafeError(
      new Error(
        'All configured providers failed. gemini/gemini-flash-latest: Gemini API 429 RESOURCE_EXHAUSTED | openrouter/anthropic/claude-sonnet-4: OpenRouter API 402 requires more credits',
      ),
      "verification",
    );
    expect(out.message).toMatch(/every ai provider is unavailable/i);
    expect(out.message).toMatch(/gemini.{0,30}quota is exhausted/i);
    expect(out.message).toMatch(/openrouter.{0,30}out of credit/i);
  });

  it("still leaks no infrastructure detail in the combined message", () => {
    const out = clientSafeError(
      new Error("All configured providers failed. gemini: key AIzaSyC7xxxxxxxxxx | openrouter: https://xyz.supabase.co"),
      "verification",
    );
    expect(out.message).not.toMatch(/AIza|supabase\.co/);
  });
});

describe("per-provider cause attribution", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it("reports a depleted Hugging Face balance as billing, not a bad model id", () => {
    const out = clientSafeError(
      new Error(
        'All configured providers failed. gemini/gemini-flash-latest: Gemini API 429 RESOURCE_EXHAUSTED | huggingface/Qwen/Qwen2.5-VL-72B-Instruct: Hugging Face API 402: {"error":"You have depleted your monthly included credits."}',
      ),
      "verification",
    );
    expect(out.message).toMatch(/hugging face.{0,40}out of credit/i);
    expect(out.message).not.toMatch(/model id/i);
  });

  it("reports an unservable Hugging Face model id as a model problem", () => {
    const out = clientSafeError(
      new Error(
        'All configured providers failed. gemini/gemini-flash-latest: Gemini API 429 RESOURCE_EXHAUSTED | huggingface/Qwen/Qwen2.5-VL-7B-Instruct: Hugging Face API 400: model_not_supported',
      ),
      "verification",
    );
    expect(out.message).toMatch(/model id is not available/i);
    expect(out.message).toMatch(/HUGGINGFACE_MODEL/);
  });

  it("explains OpenRouter's free-account prompt cap", () => {
    const out = clientSafeError(
      new Error("OpenRouter API 402: Prompt tokens limit exceeded: 1611 > 1541"),
      "verification",
    );
    expect(out.message).toMatch(/prompt/i);
    expect(out.message).toMatch(/credits/i);
    expect(out.message).not.toMatch(/Reference/);
  });
});
