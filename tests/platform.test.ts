import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { parseJson } from "../api/_gemini.js";
import { rateLimit, callerKey } from "../api/_ratelimit.js";
import { isUuid } from "../api/_identity.js";

/**
 * Regression suite for the envelope-drift bug.
 *
 * Live output returned a bare JSON array where the schema documented an object. The content was
 * correct every time; a strict parser threw it away and every claim collapsed to needs_review.
 */
describe("parseJson", () => {
  it("parses plain JSON and markdown-fenced JSON", () => {
    expect(parseJson<{ a: number }>('{"a":1}').a).toBe(1);
    expect(parseJson<{ a: number }>('```json\n{"a":1}\n```').a).toBe(1);
    expect(parseJson<{ a: number }>('```\n{"a":1}\n```').a).toBe(1);
  });

  it("recovers an object buried in prose", () => {
    expect(parseJson<{ a: number }>('Sure! Here you go:\n{"a":1}\nHope that helps.').a).toBe(1);
  });

  it("recovers a bare array — the shape that actually shipped broken", () => {
    const rows = parseJson<Array<{ field: string }>>('```json\n[{"field":"Total"}]\n```');
    expect(Array.isArray(rows)).toBe(true);
    expect(rows[0].field).toBe("Total");
  });

  it("throws a diagnosable error on unparseable output instead of a bare SyntaxError", () => {
    expect(() => parseJson("the model refused to answer")).toThrow(/unparseable JSON/i);
  });
});

describe("rate limiter", () => {
  // Each test uses a distinct key so windows never bleed between cases.
  it("allows up to the limit then blocks with a retry hint", () => {
    const key = `test-${Math.random()}`;
    for (let i = 0; i < 3; i++) expect(rateLimit(key, 3, 60_000).allowed).toBe(true);
    const blocked = rateLimit(key, 3, 60_000);
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
  });

  it("reports remaining budget accurately", () => {
    const key = `test-${Math.random()}`;
    expect(rateLimit(key, 5, 60_000).remaining).toBe(4);
    expect(rateLimit(key, 5, 60_000).remaining).toBe(3);
  });

  it("resets once the window elapses", async () => {
    const key = `test-${Math.random()}`;
    expect(rateLimit(key, 1, 20).allowed).toBe(true);
    expect(rateLimit(key, 1, 20).allowed).toBe(false); // still inside the window
    await new Promise((r) => setTimeout(r, 30)); // wait it out for real
    expect(rateLimit(key, 1, 20).allowed).toBe(true);
  });

  it("isolates callers from one another", () => {
    const a = `a-${Math.random()}`, b = `b-${Math.random()}`;
    rateLimit(a, 1, 60_000);
    expect(rateLimit(a, 1, 60_000).allowed).toBe(false);
    expect(rateLimit(b, 1, 60_000).allowed).toBe(true);
  });

  it("keys by workspace when known, falling back to forwarded IP", () => {
    expect(callerKey({ headers: {} }, "ws-1")).toBe("user:ws-1");
    expect(callerKey({ headers: { "x-forwarded-for": "203.0.113.5, 10.0.0.1" } })).toBe("ip:203.0.113.5");
    expect(callerKey({ headers: {}, socket: {} })).toBe("ip:unknown");
  });
});

describe("uuid validation", () => {
  it("accepts real UUIDs and rejects the placeholder ids that broke persistence", () => {
    expect(isUuid("bfa82daf-1f1e-4d3a-9c2b-0a1b2c3d4e5f")).toBe(true);
    // These are exactly the ids that used to violate the foreign key.
    expect(isUuid("claim-1")).toBe(false);
    expect(isUuid("")).toBe(false);
    expect(isUuid(null)).toBe(false);
    expect(isUuid("../../etc/passwd")).toBe(false);
  });

  it("rejects PostgREST filter injection attempts", () => {
    // A client-supplied id is interpolated into a PostgREST filter; anything but a UUID is refused.
    expect(isUuid("1&or=(status.eq.verified)")).toBe(false);
    expect(isUuid("*")).toBe(false);
  });
});

/**
 * Failover must walk MODELS, not just providers.
 *
 * Gemini bills its free-tier daily allowance per model (quotaId
 * `GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Measured against a live key:
 * gemini-flash-latest returned 429 while gemini-flash-lite-latest answered normally at the same
 * moment. A one-model-per-provider chain gave up on Gemini at that first 429 and failed the whole
 * request while a working model sat unused.
 */
describe("resolutionChain", () => {
  const saved = { ...process.env };
  beforeEach(() => {
    process.env.GEMINI_API_KEY = "test-gemini";
    process.env.OPENROUTER_API_KEY = "test-openrouter";
    process.env.HUGGINGFACE_API_KEY = "test-hf";
  });
  afterEach(() => {
    process.env = { ...saved };
  });

  it("offers a provider's sibling models before moving to another vendor", async () => {
    const { resolutionChain } = await import("../api/_providers/index.js");
    const chain = resolutionChain().map((c) => `${c.providerId}/${c.model}`);

    const geminiModels = chain.filter((c) => c.startsWith("gemini/"));
    expect(geminiModels.length).toBeGreaterThan(1);
    expect(geminiModels.some((m) => m.includes("flash-lite"))).toBe(true);

    // Every Gemini candidate must precede the first non-Gemini one.
    const lastGemini = chain.map((c) => c.startsWith("gemini/")).lastIndexOf(true);
    const firstOther = chain.findIndex((c) => !c.startsWith("gemini/"));
    expect(lastGemini).toBeLessThan(firstOther);
  });

  it("never repeats a provider/model pair", async () => {
    const { resolutionChain } = await import("../api/_providers/index.js");
    const chain = resolutionChain().map((c) => `${c.providerId}/${c.model}`);
    expect(new Set(chain).size).toBe(chain.length);
  });

  it("promotes an explicitly requested provider and model to the front", async () => {
    const { resolutionChain } = await import("../api/_providers/index.js");
    const chain = resolutionChain("gemini", "gemini-flash-lite-latest");
    expect(`${chain[0].providerId}/${chain[0].model}`).toBe("gemini/gemini-flash-lite-latest");
  });

  it("skips providers whose key is absent", async () => {
    delete process.env.OPENROUTER_API_KEY;
    const { resolutionChain } = await import("../api/_providers/index.js");
    expect(resolutionChain().some((c) => c.providerId === "openrouter")).toBe(false);
  });
});
