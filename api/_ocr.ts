/**
 * OCR engine abstraction.
 *
 * Reading text off a page is a deterministic task and should not consume a language model. Until
 * now the pipeline asked a vision model to transcribe, which was slow (13s on a one-page invoice),
 * non-reproducible run to run, and burned quota on work that needs no reasoning.
 *
 * PaddleOCR is the primary engine. It is a Python library, so it cannot execute inside a Node
 * serverless function — it runs as a separate service (see python/server.py) and is reached over
 * HTTP via OCR_SERVICE_URL. When that service is unreachable the pipeline falls back to model
 * transcription so the product still works, and the response says which engine actually ran.
 *
 * The engine only ever reports what it read. It never reasons, never infers, never fills gaps.
 */
import type { RawTextBlock, TranscriptionResult, DocumentPayload } from "./_providers/types.js";

export type OcrEngineId = "paddleocr" | "model-transcription";

export interface OcrOutcome extends TranscriptionResult {
  engine: OcrEngineId;
  /** Populated when the primary engine was unavailable and the fallback ran. */
  degradedReason?: string;
}

const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL;
const OCR_TIMEOUT_MS = Number(process.env.OCR_TIMEOUT_MS || 30_000);

export const ocrConfigured = (): boolean => Boolean(OCR_SERVICE_URL);

interface PaddleWord {
  text: string;
  /** Quadrilateral in pixels: [[x,y] x4] as PaddleOCR emits it. */
  box: number[][];
  confidence: number;
  page: number;
}

interface PaddleResponse {
  pages: number;
  width: number;
  height: number;
  words: PaddleWord[];
}

/**
 * Run PaddleOCR over a document.
 *
 * Throws when the service is unreachable or errors, so the caller can decide whether to degrade.
 * It deliberately does not swallow failures: silently producing an empty index would look like a
 * blank document rather than a broken dependency.
 */
export async function runPaddleOcr(document: DocumentPayload): Promise<OcrOutcome> {
  if (!OCR_SERVICE_URL) throw new Error("OCR_SERVICE_URL is not configured");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), OCR_TIMEOUT_MS);
  try {
    const response = await fetch(`${OCR_SERVICE_URL.replace(/\/$/, "")}/ocr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: document.data, mimeType: document.mimeType, fileName: document.fileName }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`OCR service ${response.status}: ${(await response.text()).slice(0, 300)}`);
    const parsed = (await response.json()) as PaddleResponse;
    return { ...toTranscription(parsed), engine: "paddleocr" };
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`OCR service timed out after ${OCR_TIMEOUT_MS}ms`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Convert word-level OCR output into the block-level index the retrieval engine expects.
 *
 * Words are grouped into lines by vertical overlap, then lines into blocks by proximity, because
 * retrieval matches phrases ("Payment Terms: Net 30") not isolated tokens. Coordinates are
 * converted from pixels to the 0-1000 normalised space the rest of the pipeline already speaks,
 * so no downstream code changes.
 */
function toTranscription(result: PaddleResponse): TranscriptionResult {
  const blocks: RawTextBlock[] = [];

  for (let page = 1; page <= Math.max(1, result.pages); page++) {
    const words = result.words.filter((w) => (w.page || 1) === page && w.text?.trim());
    if (words.length === 0) continue;

    const boxed = words.map((word) => {
      const xs = word.box.map((point) => point[0]);
      const ys = word.box.map((point) => point[1]);
      return {
        text: word.text.trim(),
        confidence: word.confidence,
        left: Math.min(...xs),
        right: Math.max(...xs),
        top: Math.min(...ys),
        bottom: Math.max(...ys),
      };
    });

    // Group into lines: words whose vertical centres sit within half a line height of each other.
    const lines: (typeof boxed)[] = [];
    for (const word of boxed.sort((a, b) => a.top - b.top || a.left - b.left)) {
      const height = word.bottom - word.top;
      const line = lines.find((candidate) => {
        const centre = candidate.reduce((sum, w) => sum + (w.top + w.bottom) / 2, 0) / candidate.length;
        return Math.abs((word.top + word.bottom) / 2 - centre) < Math.max(height * 0.6, 4);
      });
      if (line) line.push(word);
      else lines.push([word]);
    }

    for (const line of lines) {
      const ordered = line.sort((a, b) => a.left - b.left);
      const text = ordered.map((w) => w.text).join(" ").replace(/\s+/g, " ").trim();
      if (!text) continue;

      const left = Math.min(...ordered.map((w) => w.left));
      const right = Math.max(...ordered.map((w) => w.right));
      const top = Math.min(...ordered.map((w) => w.top));
      const bottom = Math.max(...ordered.map((w) => w.bottom));

      // Mean word confidence becomes block legibility — a real measurement, not an estimate.
      const legibility = Math.round((ordered.reduce((sum, w) => sum + w.confidence, 0) / ordered.length) * 100);

      blocks.push({
        page,
        text,
        legibility: Math.max(0, Math.min(100, legibility)),
        region: inferRegion(top, bottom, result.height),
        // Normalised [ymin, xmin, ymax, xmax] 0-1000 — the convention _geometry already handles.
        box_2d: [
          Math.round((top / result.height) * 1000),
          Math.round((left / result.width) * 1000),
          Math.round((bottom / result.height) * 1000),
          Math.round((right / result.width) * 1000),
        ],
      });
    }
  }

  return {
    // OCR reads text; it does not classify documents. Classification is inferred later from
    // content by the reasoning pass, or left unknown. Guessing here would be fabrication.
    documentType: "Unknown document",
    pageCount: Math.max(1, result.pages),
    blocks,
  };
}

/**
 * Geometric region only — top band is header, bottom band is footer, the rest is body.
 * No content-based guessing: a deterministic engine must not infer "signature" from a squiggle.
 */
function inferRegion(top: number, bottom: number, pageHeight: number): string {
  if (pageHeight <= 0) return "body";
  const centre = (top + bottom) / 2 / pageHeight;
  if (centre < 0.12) return "header";
  if (centre > 0.88) return "footer";
  return "body";
}
