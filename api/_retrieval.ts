/**
 * Evidence Retrieval Engine.
 *
 * An independent service between claim extraction and verification. It searches the document
 * index with several orthogonal strategies and returns ranked candidates. Crucially it runs
 * BEFORE the verifying model sees the claim, and the model may only cite what this engine
 * returned — so evidence is retrieved, not asserted by the same model that is being checked.
 *
 * Every strategy that fires is recorded on the candidate, which is what lets the UI say
 * truthfully which surfaces were searched for a given claim.
 */
import { distance, type BoundingBox } from "./_geometry.js";
import { tokenize, type DocumentIndex, type TextBlock } from "./_documentIndex.js";
import type { RegionKind } from "./_providers/types.js";

export type RetrievalStrategy =
  | "value-match"
  | "lexical"
  | "numeric"
  | "field-label"
  | "region-affinity"
  | "spatial-neighbour";

export interface RetrievedCandidate {
  block: TextBlock;
  /** 0-1 blended retrieval score. */
  score: number;
  /** 0-1 best textual similarity between the claimed value and this block. */
  valueSimilarity: number;
  strategies: RetrievalStrategy[];
}

export interface RetrievalReport {
  candidates: RetrievedCandidate[];
  /** Every surface the engine searched, whether or not it produced a hit. */
  searched: string[];
  strategiesHit: RetrievalStrategy[];
}

const MAX_CANDIDATES = 6;
const MIN_SCORE = 0.12;

/* ── Similarity primitives ───────────────────────────────────────────────── */

/** Sørensen–Dice over character bigrams: robust to OCR noise and word order. */
export function diceSimilarity(a: string, b: string): number {
  const left = bigrams(a.toLowerCase());
  const right = bigrams(b.toLowerCase());
  if (left.size === 0 || right.size === 0) return a.toLowerCase() === b.toLowerCase() ? 1 : 0;
  let shared = 0;
  for (const gram of left) if (right.has(gram)) shared += 1;
  return (2 * shared) / (left.size + right.size);
}

function bigrams(text: string): Set<string> {
  const clean = text.replace(/\s+/g, " ").trim();
  const grams = new Set<string>();
  for (let i = 0; i < clean.length - 1; i++) grams.add(clean.slice(i, i + 2));
  return grams;
}

/** Jaccard over content tokens: captures semantic overlap the character view misses. */
export function tokenOverlap(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  const shared = new Set(a.filter((token) => setB.has(token)));
  return shared.size / new Set([...a, ...b]).size;
}

/** Best similarity between the value and any window of the block, so a long block isn't penalised. */
function containmentSimilarity(value: string, block: TextBlock): number {
  const haystack = block.text.toLowerCase();
  const needle = value.toLowerCase().trim();
  if (needle.length === 0) return 0;
  if (haystack.includes(needle)) return 1;

  const whole = diceSimilarity(needle, block.text);
  // Slide a window the length of the value across the block.
  let best = whole;
  const window = Math.max(needle.length, 4);
  for (let i = 0; i + window <= haystack.length; i += Math.max(1, Math.floor(window / 3))) {
    best = Math.max(best, diceSimilarity(needle, haystack.slice(i, i + window)));
  }
  return best;
}

/* ── Numeric / date extraction ───────────────────────────────────────────── */

const NUMERIC_RE = /\d[\d,.\-/]*\d|\d/g;

function numericTokens(text: string): string[] {
  return (text.match(NUMERIC_RE) || []).map((token) => token.replace(/[,\s]/g, "")).filter((token) => token.length > 0);
}

function numericMatch(valueNumbers: string[], block: TextBlock): number {
  if (valueNumbers.length === 0) return 0;
  const blockNumbers = numericTokens(block.text);
  if (blockNumbers.length === 0) return 0;
  const hits = valueNumbers.filter((n) => blockNumbers.some((b) => b === n || b.endsWith(n) || n.endsWith(b)));
  return hits.length / valueNumbers.length;
}

/* ── Region affinity ─────────────────────────────────────────────────────── */

/**
 * Which page regions plausibly carry a given kind of fact. This is a retrieval prior over
 * generic document geography, not a document-type schema — no field name is hardcoded, the
 * hints are matched against whatever field the caller supplied.
 */
const REGION_HINTS: Array<{ pattern: RegExp; regions: RegionKind[] }> = [
  { pattern: /\b(total|subtotal|amount|tax|gst|vat|qty|quantity|rate|price|line item|balance)\b/i, regions: ["table", "footer"] },
  { pattern: /\b(name|title|company|vendor|supplier|issuer|organisation|organization|employer|hospital|university)\b/i, regions: ["header", "logo"] },
  { pattern: /\b(sign|signature|signed|authorised|authorized|witness)\b/i, regions: ["signature", "footer"] },
  { pattern: /\b(page|footer|disclaimer|terms|conditions|notice)\b/i, regions: ["footer"] },
  { pattern: /\b(figure|chart|diagram|graph|image|photo)\b/i, regions: ["figure"] },
];

function regionAffinity(field: string, block: TextBlock): number {
  for (const hint of REGION_HINTS) {
    if (hint.pattern.test(field)) return hint.regions.includes(block.region) ? 1 : 0;
  }
  return 0;
}

/* ── The engine ──────────────────────────────────────────────────────────── */

export function retrieveEvidence(index: DocumentIndex, field: string, value: string): RetrievalReport {
  const searched = surfacesSearched(index);
  if (index.blocks.length === 0) return { candidates: [], searched, strategiesHit: [] };

  const fieldTokens = tokenize(field);
  const valueTokens = tokenize(value);
  const queryTokens = [...new Set([...fieldTokens, ...valueTokens])];
  const valueNumbers = numericTokens(value);

  const scored = index.blocks.map((block) => {
    const strategies: RetrievalStrategy[] = [];

    const valueSimilarity = containmentSimilarity(value, block);
    if (valueSimilarity >= 0.55) strategies.push("value-match");

    const lexical = tokenOverlap(queryTokens, block.tokens);
    if (lexical >= 0.08) strategies.push("lexical");

    const numeric = numericMatch(valueNumbers, block);
    if (numeric > 0) strategies.push("numeric");

    // A label/value pair: the block names the field even if the value sits nearby.
    const labelHit = fieldTokens.length > 0 && tokenOverlap(fieldTokens, block.tokens) >= 0.3 ? 1 : 0;
    if (labelHit) strategies.push("field-label");

    const affinity = regionAffinity(field, block);
    if (affinity) strategies.push("region-affinity");

    const score =
      valueSimilarity * 0.45 +
      lexical * 0.2 +
      numeric * 0.15 +
      labelHit * 0.12 +
      affinity * 0.08;

    return { block, score, valueSimilarity, strategies };
  });

  // Similar-region expansion: a label block often sits next to the block holding the value,
  // so pull in close neighbours of the strongest hit that plain text search would miss.
  const anchor = [...scored].sort((a, b) => b.score - a.score)[0];
  if (anchor && anchor.score >= MIN_SCORE && anchor.block.boundingBox) {
    for (const candidate of scored) {
      if (candidate === anchor || candidate.block.page !== anchor.block.page) continue;
      if (!candidate.block.boundingBox) continue;
      const gap = distance(anchor.block.boundingBox as BoundingBox, candidate.block.boundingBox as BoundingBox);
      if (gap <= 12 && candidate.score < anchor.score) {
        candidate.score = Math.max(candidate.score, anchor.score * 0.45 * (1 - gap / 12));
        if (!candidate.strategies.includes("spatial-neighbour")) candidate.strategies.push("spatial-neighbour");
      }
    }
  }

  const candidates = scored
    .filter((candidate) => candidate.score >= MIN_SCORE && candidate.strategies.length > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_CANDIDATES)
    .map((candidate) => ({ ...candidate, score: round3(candidate.score), valueSimilarity: round3(candidate.valueSimilarity) }));

  return {
    candidates,
    searched,
    strategiesHit: [...new Set(candidates.flatMap((candidate) => candidate.strategies))],
  };
}

/** Honest inventory of what this document actually offered to search. */
function surfacesSearched(index: DocumentIndex): string[] {
  const regions = new Set(index.blocks.map((block) => block.region));
  const surfaces = ["Transcribed text", "Token index"];
  if (regions.has("header")) surfaces.push("Headers");
  if (regions.has("table")) surfaces.push("Tables");
  if (regions.has("footer")) surfaces.push("Footers");
  if (regions.has("signature")) surfaces.push("Signatures");
  if (regions.has("logo")) surfaces.push("Logos");
  if (regions.has("figure")) surfaces.push("Figures");
  if (index.quality.boundingBoxCoverage > 0) surfaces.push("Bounding boxes");
  if (index.quality.pageCount > 1) surfaces.push(`${index.quality.pageCount} pages`);
  return surfaces;
}

const round3 = (value: number) => Math.round(value * 1000) / 1000;
