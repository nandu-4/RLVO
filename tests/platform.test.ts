import { describe, it, expect } from "vitest";
import { parseJson } from "../api/_gemini.js";
import { rateLimit, callerKey } from "../api/_ratelimit.js";
import { isUuid } from "../api/_workspace.js";

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
