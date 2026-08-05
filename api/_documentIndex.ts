/**
 * The searchable document index — the substrate the Evidence Retrieval Engine searches.
 *
 * Built from a claim-blind transcription pass, so the blocks in it exist independently of any
 * claim anyone wants to prove. Container junk (PDF internals, TeX producer strings, binary
 * stream fragments) is rejected here, once, rather than being filtered at every consumer.
 */
import { area, normalizeBoundingBox, type BoundingBox } from "./_geometry.js";
import type { RawTextBlock, RegionKind, TranscriptionResult } from "./_providers/types.js";

export interface TextBlock {
  id: string;
  page: number;
  text: string;
  /** Lowercased, punctuation-stripped token list, precomputed for retrieval. */
  tokens: string[];
  boundingBox?: BoundingBox;
  region: RegionKind;
  /** 0-100 confidence that this block was read correctly off the page. */
  legibility: number;
}

export interface IndexQuality {
  blockCount: number;
  pageCount: number;
  meanLegibility: number;
  /** Share of blocks the provider read with low confidence. */
  lowLegibilityRatio: number;
  /** Share of blocks that carry usable coordinates. */
  boundingBoxCoverage: number;
  /** Share of blocks whose page footprint is tiny — a proxy for small or dense type. */
  smallTypeRatio: number;
}

export interface DocumentIndex {
  documentType: string;
  blocks: TextBlock[];
  quality: IndexQuality;
}

const CONTAINER_JUNK =
  /\b(pdftex|pdflatex|xetex|luatex|tex live|ghostscript|skia\/pdf|microsoft. print to pdf|xref|endstream|endobj|startxref|\/type\s*\/page|\/producer|\/creationdate|trailer|linearized)\b|\b\d+\s+\d+\s+obj\b|^%%?(pdf|eof)/i;

const VALID_REGIONS: RegionKind[] = ["header", "body", "table", "footer", "signature", "logo", "figure"];

const STOPWORDS = new Set([
  "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "at", "by", "is", "are", "was", "were",
  "this", "that", "with", "as", "from", "be", "it", "its",
]);

/** Shared tokenizer: retrieval and scoring must agree on what a token is. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}.,/:%-]+/gu, " ")
    .split(/\s+/)
    .map((token) => token.replace(/^[.,:/-]+|[.,:/-]+$/g, ""))
    .filter((token) => token.length > 0 && !STOPWORDS.has(token));
}

const clamp100 = (value: unknown, fallback: number) => {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.min(100, Math.round(n))) : fallback;
};

export function buildDocumentIndex(transcription: TranscriptionResult): DocumentIndex {
  const blocks: TextBlock[] = [];

  for (const raw of transcription.blocks) {
    const block = normalizeBlock(raw, blocks.length);
    if (block) blocks.push(block);
  }

  const deduped = dedupe(blocks);
  return {
    documentType: transcription.documentType || "Unknown document",
    blocks: deduped,
    quality: measureQuality(deduped, transcription.pageCount),
  };
}

function normalizeBlock(raw: RawTextBlock, index: number): TextBlock | null {
  if (!raw || typeof raw.text !== "string") return null;
  const text = raw.text.replace(/\s+/g, " ").trim();
  if (text.length === 0 || CONTAINER_JUNK.test(text)) return null;

  const page = Number.isInteger(raw.page) && raw.page > 0 ? raw.page : 1;
  const region = VALID_REGIONS.includes(raw.region as RegionKind) ? (raw.region as RegionKind) : "body";

  return {
    id: `block-${index + 1}`,
    page,
    text,
    tokens: tokenize(text),
    // Accept either key: box_2d is Gemini's native trained format and is what the prompt now
    // asks for, but other adapters may emit boundingBox.
    boundingBox: normalizeBoundingBox(raw.box_2d ?? raw.boundingBox),
    region,
    // Absent legibility is treated as moderate, not perfect: an unstated confidence is not a
    // high one, and defaulting to 100 would silently inflate every downstream score.
    legibility: clamp100(raw.legibility, 60),
  };
}

/** Providers repeat blocks across overlapping crops; keep the most legible copy of each. */
function dedupe(blocks: TextBlock[]): TextBlock[] {
  const seen = new Map<string, TextBlock>();
  for (const block of blocks) {
    const key = `${block.page}::${block.text.toLowerCase()}`;
    const existing = seen.get(key);
    if (!existing || block.legibility > existing.legibility) seen.set(key, block);
  }
  return [...seen.values()].map((block, index) => ({ ...block, id: `block-${index + 1}` }));
}

function measureQuality(blocks: TextBlock[], declaredPages: number): IndexQuality {
  if (blocks.length === 0) {
    return { blockCount: 0, pageCount: Math.max(1, declaredPages), meanLegibility: 0, lowLegibilityRatio: 1, boundingBoxCoverage: 0, smallTypeRatio: 0 };
  }
  const pages = new Set(blocks.map((block) => block.page));
  const withBox = blocks.filter((block) => block.boundingBox);
  const smallType = withBox.filter((block) => area(block.boundingBox as BoundingBox) < 0.35);

  return {
    blockCount: blocks.length,
    pageCount: Math.max(pages.size, declaredPages || 1),
    meanLegibility: Math.round(blocks.reduce((sum, block) => sum + block.legibility, 0) / blocks.length),
    lowLegibilityRatio: round2(blocks.filter((block) => block.legibility < 65).length / blocks.length),
    boundingBoxCoverage: round2(withBox.length / blocks.length),
    smallTypeRatio: withBox.length > 0 ? round2(smallType.length / withBox.length) : 0,
  };
}

const round2 = (value: number) => Math.round(value * 100) / 100;
