import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  contentHash,
  deviceIdFrom,
  deleteSession,
  findByHash,
  listSessions,
  localStoreEnabled,
  readSession,
  saveSession,
  storageDurability,
  type SessionSummary,
} from "../api/_localstore.js";

/**
 * The local store is what makes results persist on a deployment with no database. Before it, the
 * entire replay feature — snapshots, history, no-model replay — was built but unreachable, and
 * every verification was silently discarded.
 */

let dir: string;
const saved = { ...process.env };

const summary = (over: Partial<SessionSummary> = {}): SessionSummary => ({
  id: "11111111-1111-4111-8111-111111111111",
  createdAt: "2026-08-05T00:00:00.000Z",
  fileName: "invoice.png",
  documentType: "Invoice",
  provider: "gemini",
  model: "gemini-flash-lite-latest",
  trustScore: 66,
  verificationMode: "cross-check",
  ...over,
});

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "truthlens-test-"));
  process.env.TRUTHLENS_DATA_DIR = dir;
  delete process.env.TRUTHLENS_LOCAL_STORE;
});

afterEach(async () => {
  process.env = { ...saved };
  await rm(dir, { recursive: true, force: true });
});

describe("content addressing", () => {
  const claims = [
    { field: "Vendor", value: "Google" },
    { field: "Total", value: "$13,511.00" },
  ];

  it("is stable across claim order and incidental casing/whitespace", () => {
    const a = contentHash("DOC", claims, "cross-check");
    const b = contentHash("DOC", [...claims].reverse(), "cross-check");
    const c = contentHash("DOC", [{ field: " vendor ", value: "google" }, { field: "Total", value: "$13,511.00" }], "cross-check");
    expect(a).toBe(b);
    expect(a).toBe(c);
  });

  it("separates different documents, claims and modes", () => {
    const base = contentHash("DOC", claims, "cross-check");
    expect(contentHash("OTHER", claims, "cross-check")).not.toBe(base);
    expect(contentHash("DOC", [{ field: "Vendor", value: "Oracle" }], "cross-check")).not.toBe(base);
    expect(contentHash("DOC", claims, "self-check")).not.toBe(base);
  });

  /*
   * A field/value pair must not be able to impersonate a different one by smuggling the
   * separator: {field:"a=b", value:"c"} and {field:"a", value:"b=c"} are different questions.
   */
  it("does not collide when a claim contains the separator characters", () => {
    const x = contentHash("DOC", [{ field: "a=b", value: "c" }], "cross-check");
    const y = contentHash("DOC", [{ field: "a", value: "b=c" }], "cross-check");
    expect(x).not.toBe(y);
  });
});

describe("device identity", () => {
  it("accepts a well-formed device id", () => {
    expect(deviceIdFrom({ headers: { "x-truthlens-device": "abcdef0123456789" } })).toBe("abcdef0123456789");
  });

  /*
   * The device id is attacker-controlled and becomes a path segment. Anything that could escape
   * the store root must collapse to the shared bucket rather than resolve.
   */
  it("refuses path traversal and other unsafe ids", () => {
    for (const hostile of ["../../etc/passwd", "..", "a/../../b", "short", "", "with space", "x".repeat(65)]) {
      expect(deviceIdFrom({ headers: { "x-truthlens-device": hostile } })).toBe("shared-local");
    }
  });

  it("falls back to a shared bucket when the header is absent", () => {
    expect(deviceIdFrom({ headers: {} })).toBe("shared-local");
    expect(deviceIdFrom({})).toBe("shared-local");
  });
});

describe("session round trip", () => {
  const device = "device0123456789";

  it("stores and returns the COMPLETE snapshot, not a summary of it", async () => {
    const snapshot = {
      id: summary().id,
      claims: [{ field: "Vendor", status: "corrected", evidence: [{ text: "Oracle", boundingBox: { x: 1, y: 2, width: 3, height: 4 }, pageNumber: 1 }] }],
      textBlocks: [{ text: "Oracle Corporation", page: 1, box_2d: [1, 2, 3, 4] }],
      relations: [{ from: "claim-1", to: "claim-2", kind: "same-region" }],
      timeline: [{ step: "intake", durationMs: 12 }],
      summary: { trustScore: 66 },
      ocr: { engine: "model-transcription" },
      provider: "gemini",
      modelUsed: "gemini-flash-lite-latest",
      createdAt: "2026-08-05T00:00:00.000Z",
    };
    await saveSession(device, snapshot, summary());

    const read = (await readSession(device, summary().id)) as typeof snapshot | null;
    // Byte-identical: a replay must not be a reconstruction.
    expect(read).toEqual(snapshot);
    expect(read?.claims[0].evidence[0].boundingBox).toEqual({ x: 1, y: 2, width: 3, height: 4 });
    expect(read?.textBlocks).toHaveLength(1);
  });

  it("lists newest first and finds a repeat by content hash", async () => {
    const hash = contentHash("DOC", [{ field: "Vendor", value: "Google" }], "cross-check");
    await saveSession(device, { a: 1 }, summary({ id: "aaaaaaaa-1111-4111-8111-111111111111", contentHash: hash }));
    await saveSession(device, { b: 2 }, summary({ id: "bbbbbbbb-2222-4222-8222-222222222222" }));

    const list = await listSessions(device);
    expect(list[0].id).toBe("bbbbbbbb-2222-4222-8222-222222222222");
    expect(list).toHaveLength(2);

    const found = await findByHash(device, hash);
    expect(found?.id).toBe("aaaaaaaa-1111-4111-8111-111111111111");
    expect(await findByHash(device, "no-such-hash")).toBeNull();
  });

  it("keeps devices isolated from each other", async () => {
    await saveSession("device0123456789", { a: 1 }, summary());
    expect(await listSessions("otherdevice12345")).toHaveLength(0);
    expect(await readSession("otherdevice12345", summary().id)).toBeNull();
  });

  it("replaces rather than duplicates when the same id is written twice", async () => {
    await saveSession(device, { v: 1 }, summary());
    await saveSession(device, { v: 2 }, summary());
    expect(await listSessions(device)).toHaveLength(1);
    expect(await readSession(device, summary().id)).toEqual({ v: 2 });
  });

  it("deletes a session and its snapshot", async () => {
    await saveSession(device, { a: 1 }, summary());
    expect(await deleteSession(device, summary().id)).toBe(true);
    expect(await listSessions(device)).toHaveLength(0);
    expect(await readSession(device, summary().id)).toBeNull();
    expect(await deleteSession(device, summary().id)).toBe(false);
  });

  it("returns null for a missing session instead of throwing", async () => {
    expect(await readSession(device, "99999999-9999-4999-8999-999999999999")).toBeNull();
    expect(await listSessions("neverwritten1234")).toEqual([]);
  });
});

describe("store configuration", () => {
  it("is on by default and switchable off", () => {
    expect(localStoreEnabled()).toBe(true);
    process.env.TRUTHLENS_LOCAL_STORE = "off";
    expect(localStoreEnabled()).toBe(false);
  });

  /* An explicit data dir is durable anywhere; Vercel's /tmp is not, and must not claim to be. */
  it("reports durability honestly", () => {
    expect(storageDurability()).toBe("durable");
    delete process.env.TRUTHLENS_DATA_DIR;
    process.env.VERCEL = "1";
    expect(storageDurability()).toBe("ephemeral");
  });
});
