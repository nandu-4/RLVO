import { describe, it, expect } from "vitest";
import { resolveMediaType, SUPPORTED_MIME } from "../api/_media.js";

const b64 = (bytes: number[]) => Buffer.from(new Uint8Array([...bytes, ...Array(32).fill(0)])).toString("base64");
const JPEG = b64([0xff, 0xd8, 0xff, 0xe0]);
const PNG = b64([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const PDF = b64([0x25, 0x50, 0x44, 0x46, 0x2d, 0x31]);
const WEBP = b64([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4, 0x57, 0x45, 0x42, 0x50]);

/**
 * Regression suite for a real failure: a PNG saved as ".jpeg" was declared image/jpeg by the
 * browser, and Anthropic rejected it with a bare "Could not process image" that was impossible
 * to diagnose from the response.
 */
describe("resolveMediaType", () => {
  it("detects each supported format from its signature", () => {
    expect(resolveMediaType(JPEG, "", "x").mimeType).toBe("image/jpeg");
    expect(resolveMediaType(PNG, "", "x").mimeType).toBe("image/png");
    expect(resolveMediaType(PDF, "", "x").mimeType).toBe("application/pdf");
    expect(resolveMediaType(WEBP, "", "x").mimeType).toBe("image/webp");
  });

  it("corrects a PNG mislabelled as JPEG — the case that broke live", () => {
    const media = resolveMediaType(PNG, "image/jpeg", "contractor-invoice - Copy.jpeg");
    expect(media.mimeType).toBe("image/png");
    expect(media.corrected).toBe(true);
    expect(media.declared).toBe("image/jpeg");
  });

  it("corrects a PDF renamed with an image extension", () => {
    expect(resolveMediaType(PDF, "image/jpeg", "scan.jpg").mimeType).toBe("application/pdf");
  });

  it("does not flag a correctly labelled file", () => {
    expect(resolveMediaType(JPEG, "image/jpeg", "a.jpg").corrected).toBe(false);
  });

  it("falls back to the declared type when the signature is unknown", () => {
    const media = resolveMediaType(b64([0x00, 0x01, 0x02, 0x03]), "image/jpeg", "a.jpg");
    expect(media.mimeType).toBe("image/jpeg");
    expect(media.corrected).toBe(false);
  });

  it("infers PDF from the extension when nothing else is known", () => {
    expect(resolveMediaType(b64([0, 1, 2, 3]), "", "doc.pdf").mimeType).toBe("application/pdf");
  });

  it("survives malformed base64 without throwing", () => {
    expect(() => resolveMediaType("!!!not base64!!!", "image/png", "x.png")).not.toThrow();
  });

  it("only admits formats every provider can read", () => {
    expect(SUPPORTED_MIME).toContain("application/pdf");
    expect(SUPPORTED_MIME).not.toContain("image/tiff" as never);
  });
});
