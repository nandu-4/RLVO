/**
 * Media type detection from content, not from the filename.
 *
 * Browsers derive a file's MIME type from its extension, so a PNG saved or renamed as ".jpeg"
 * arrives labelled `image/jpeg` with PNG bytes inside. Vendors validate the bytes against the
 * declared type and reject the mismatch — Anthropic returns a bare "Could not process image",
 * which is impossible to diagnose from the outside. Users rename and re-save files constantly,
 * so trusting the label is not safe.
 *
 * Magic numbers are checked against the decoded prefix only; the payload is never fully decoded
 * here, which keeps this cheap for multi-megabyte uploads.
 */

export type DetectedMime =
  | "image/jpeg"
  | "image/png"
  | "image/webp"
  | "image/gif"
  | "application/pdf";

const SIGNATURES: Array<{ mime: DetectedMime; test: (b: Uint8Array) => boolean }> = [
  { mime: "image/jpeg", test: (b) => b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff },
  { mime: "image/png", test: (b) => b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47 },
  { mime: "application/pdf", test: (b) => b[0] === 0x25 && b[1] === 0x50 && b[2] === 0x44 && b[3] === 0x46 },
  { mime: "image/gif", test: (b) => b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46 },
  {
    mime: "image/webp",
    // "RIFF" .... "WEBP"
    test: (b) =>
      b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 &&
      b[8] === 0x57 && b[9] === 0x45 && b[10] === 0x42 && b[11] === 0x50,
  },
];

/** Decode just enough of the base64 payload to read a file signature. */
function prefixBytes(base64: string, byteCount = 16): Uint8Array {
  // 4 base64 chars encode 3 bytes; take a little extra and let Buffer trim.
  const chunk = base64.slice(0, Math.ceil((byteCount / 3) * 4) + 4);
  try {
    return new Uint8Array(Buffer.from(chunk, "base64").subarray(0, byteCount));
  } catch {
    return new Uint8Array();
  }
}

export interface MediaDescriptor {
  /** The type to send upstream — detected where possible, else the declared type. */
  mimeType: string;
  /** True when the declared type disagreed with the actual bytes. */
  corrected: boolean;
  declared: string;
}

/**
 * Resolve the MIME type to send to a provider.
 * Detection wins over the declared value, because the declared value is only ever a guess
 * derived from a filename the user controls.
 */
export function resolveMediaType(base64: string, declared: string, fileName: string): MediaDescriptor {
  const bytes = prefixBytes(base64);
  const detected = SIGNATURES.find((signature) => signature.test(bytes))?.mime;

  if (!detected) {
    // Unrecognised signature: fall back to the declared type, or infer from the extension.
    const fallback = declared || (/\.pdf$/i.test(fileName) ? "application/pdf" : "image/jpeg");
    return { mimeType: fallback, corrected: false, declared: fallback };
  }

  return { mimeType: detected, corrected: detected !== declared, declared: declared || detected };
}

/** Formats every configured provider can read. Anything else is rejected with a clear message. */
export const SUPPORTED_MIME: DetectedMime[] = ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"];
