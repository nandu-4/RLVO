/**
 * Trust signal computation.
 *
 * Each signal here is measured from something different, so agreement between them is real
 * corroboration rather than one number displayed five times:
 *
 *   ocrAgreement      character-level similarity between the claimed value and the transcribed
 *                     text, weighted by how legibly that text was read
 *   semanticAgreement content-token overlap between the claim and the cited evidence
 *   layoutAgreement   whether the evidence has coordinates, sits in a plausible region, and is
 *                     spatially coherent
 *   visionAgreement   the provider's visual read of the page — the only model-supplied signal,
 *                     and the only one that can legitimately be missing
 *   evidenceStrength  retrieval ranking and legibility of what the engine actually found
 *
 * A signal that was not measured is reported as unmeasured and excluded from the weighted mean,
 * never defaulted to another signal's value.
 */
import { iou, type BoundingBox } from "./_geometry.js";
import { diceSimilarity, tokenOverlap, type RetrievedCandidate } from "./_retrieval.js";
import { tokenize, type IndexQuality, type TextBlock } from "./_documentIndex.js";

export interface SignalValue {
  value: number;
  measured: boolean;
  /** What this number was computed from, in one line. */
  basis: string;
}

export interface TrustSignals {
  ocrAgreement: SignalValue;
  visionAgreement: SignalValue;
  layoutAgreement: SignalValue;
  semanticAgreement: SignalValue;
  evidenceStrength: SignalValue;
  finalTrustScore: number;
  measuredCount: number;
  /** Plain-language reasons the score is what it is. */
  why: string[];
}

const WEIGHTS = { ocrAgreement: 0.26, visionAgreement: 0.2, layoutAgreement: 0.14, semanticAgreement: 0.2, evidenceStrength: 0.2 } as const;
const pct = (value: number) => Math.max(0, Math.min(100, Math.round(value * 100)));

export interface SignalInput {
  field: string;
  claimedValue: string;
  /** Evidence the provider cited, already resolved back to indexed blocks. */
  cited: RetrievedCandidate[];
  /** Everything retrieval offered, cited or not. */
  retrieved: RetrievedCandidate[];
  /** 0-100 from the provider, or undefined when it did not assess the page visually. */
  providerVisionAgreement?: number;
  quality: IndexQuality;
}

export function computeSignals(input: SignalInput): TrustSignals {
  const { cited, retrieved, claimedValue, field } = input;
  const evidence = cited.length > 0 ? cited : [];

  /* ── OCR agreement: does the transcribed text actually say this value? ── */
  const ocr: SignalValue = evidence.length === 0
    ? { value: 0, measured: false, basis: "No evidence was cited, so no text comparison was possible." }
    : (() => {
        const best = Math.max(...evidence.map((candidate) => candidate.valueSimilarity));
        const legibility = Math.max(...evidence.map((candidate) => candidate.block.legibility)) / 100;
        // A perfect string match read off illegible text is not strong evidence.
        const value = best * (0.6 + 0.4 * legibility);
        return {
          value: pct(value),
          measured: true,
          basis: `Best text similarity ${pct(best)}% against evidence read at ${pct(legibility)}% legibility.`,
        };
      })();

  /* ── Semantic agreement: do claim and evidence talk about the same thing? ── */
  const semantic: SignalValue = evidence.length === 0
    ? { value: 0, measured: false, basis: "No evidence was cited, so no semantic comparison was possible." }
    : (() => {
        const claimTokens = tokenize(`${field} ${claimedValue}`);
        const best = Math.max(...evidence.map((candidate) => tokenOverlap(claimTokens, candidate.block.tokens)));
        // Token overlap is inherently low for short values; rescale so it is comparable to the
        // other signals rather than permanently dragging the mean down.
        const value = Math.min(1, best * 2.2);
        return { value: pct(value), measured: true, basis: `Content-token overlap of ${pct(best)}% between the claim and the cited text.` };
      })();

  /* ── Layout agreement: is the evidence grounded on the page and coherent? ── */
  const layout: SignalValue = evidence.length === 0
    ? { value: 0, measured: false, basis: "No evidence was cited, so no layout grounding could be checked." }
    : (() => {
        const boxes = evidence.map((candidate) => candidate.block.boundingBox).filter(Boolean) as BoundingBox[];
        if (boxes.length === 0) {
          return { value: 0, measured: true, basis: "Cited evidence carries no coordinates, so it could not be grounded on the page." };
        }
        const coverage = boxes.length / evidence.length;
        const regionFit = evidence.some((candidate) => candidate.strategies.includes("region-affinity")) ? 1 : 0.7;
        // Multiple boxes describing one fact should be near each other or overlapping.
        const coherence = boxes.length < 2 ? 1 : averagePairwise(boxes);
        const samePage = new Set(evidence.map((candidate) => candidate.block.page)).size === 1 ? 1 : 0.75;
        const value = coverage * 0.4 + coherence * 0.25 + regionFit * 0.2 + samePage * 0.15;
        return {
          value: pct(value),
          measured: true,
          basis: `${boxes.length} of ${evidence.length} evidence items are grounded with coordinates on page ${evidence[0].block.page}.`,
        };
      })();

  /* ── Vision agreement: the only signal the model alone can supply ── */
  const vision: SignalValue =
    typeof input.providerVisionAgreement === "number" && Number.isFinite(input.providerVisionAgreement)
      ? {
          value: Math.max(0, Math.min(100, Math.round(input.providerVisionAgreement))),
          measured: true,
          basis: "Provider's visual assessment of the rendered page.",
        }
      : { value: 0, measured: false, basis: "The provider did not report a visual assessment; this signal is excluded from the score." };

  /* ── Evidence strength: how good was what retrieval found? ── */
  const strength: SignalValue = (() => {
    if (retrieved.length === 0) {
      return { value: 0, measured: true, basis: "The retrieval engine found no candidate evidence for this claim." };
    }
    const top = retrieved[0];
    const citedBoost = evidence.length === 0 ? 0.45 : Math.min(1, 0.75 + 0.25 * Math.min(evidence.length, 3) / 3);
    const legibility = top.block.legibility / 100;
    const value = top.score * citedBoost * (0.65 + 0.35 * legibility);
    return {
      value: pct(value),
      measured: true,
      basis: `Top retrieval score ${pct(top.score)}% from ${retrieved.length} candidate(s); ${evidence.length} cited by the verifier.`,
    };
  })();

  const signals = { ocrAgreement: ocr, visionAgreement: vision, layoutAgreement: layout, semanticAgreement: semantic, evidenceStrength: strength };

  // Weighted mean over measured signals only, renormalised so an unmeasured signal neither
  // counts as zero nor is silently replaced by another signal's value.
  const measured = Object.entries(signals).filter(([, signal]) => signal.measured) as Array<[keyof typeof WEIGHTS, SignalValue]>;
  const weightSum = measured.reduce((sum, [key]) => sum + WEIGHTS[key], 0);
  const finalTrustScore = weightSum === 0 ? 0 : Math.round(measured.reduce((sum, [key, signal]) => sum + signal.value * WEIGHTS[key], 0) / weightSum);

  return { ...signals, finalTrustScore, measuredCount: measured.length, why: explain(signals, input, finalTrustScore) };
}

function averagePairwise(boxes: BoundingBox[]): number {
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      // Either overlapping or vertically adjacent counts as coherent.
      const overlap = iou(boxes[i], boxes[j]);
      const adjacency = Math.abs(boxes[i].y - boxes[j].y) < 8 ? 0.6 : 0;
      total += Math.max(overlap, adjacency);
      pairs += 1;
    }
  }
  return pairs === 0 ? 1 : Math.min(1, total / pairs + 0.4);
}

/** Turn the measurements into the "why does this score exist" narrative the spec asks for. */
function explain(signals: Record<string, SignalValue>, input: SignalInput, finalScore: number): string[] {
  const why: string[] = [];
  const { cited, retrieved, quality } = input;

  if (cited.length === 0) {
    why.push("No evidence was cited for this claim, so no automatic decision could be justified.");
  } else {
    const pages = [...new Set(cited.map((candidate) => candidate.block.page))].sort((a, b) => a - b);
    why.push(`${cited.length} evidence item(s) cited from page ${pages.join(", ")}.`);
  }

  if (signals.ocrAgreement.measured) {
    const value = signals.ocrAgreement.value;
    why.push(
      value >= 90 ? "The transcribed text matches the claimed value almost exactly."
        : value >= 70 ? "The transcribed text closely matches the claimed value with minor differences."
        : value >= 40 ? "The transcribed text only partially matches the claimed value."
        : "The transcribed text does not support the claimed value.",
    );
  }

  if (signals.layoutAgreement.measured) {
    why.push(
      signals.layoutAgreement.value >= 75
        ? "Evidence is grounded to coordinates in a region consistent with this kind of fact."
        : signals.layoutAgreement.value >= 40
        ? "Evidence is partially grounded; some items lack coordinates or sit in an unexpected region."
        : "Evidence could not be grounded to a specific region of the page.",
    );
  }

  if (!signals.visionAgreement.measured) {
    why.push("The provider reported no independent visual assessment, so the score rests on text, layout, and retrieval signals.");
  } else if (signals.visionAgreement.value >= 85) {
    why.push("The provider's visual read of the page independently supports the claim.");
  } else if (signals.visionAgreement.value < 50) {
    why.push("The provider's visual read of the page does not clearly support the claim.");
  }

  if (retrieved.length > cited.length && cited.length > 0) {
    why.push(`${retrieved.length - cited.length} further candidate(s) were retrieved but not relied upon.`);
  }
  if (quality.lowLegibilityRatio > 0.3) {
    why.push(`${Math.round(quality.lowLegibilityRatio * 100)}% of the document was read with low confidence, which caps how strong any evidence here can be.`);
  }
  if (finalScore < 50) why.push("Overall agreement is below the threshold at which an automatic decision would be trusted.");

  return why;
}

/* ── Pre-verification hallucination risk ─────────────────────────────────── */

export interface HallucinationRisk {
  level: "LOW" | "MEDIUM" | "HIGH";
  score: number;
  reasons: string[];
}

/**
 * Predicts how likely a claim is to be hallucinated BEFORE the verifying model runs.
 *
 * This is deliberately computed from the document and the retrieval result only — never from the
 * verification outcome. A post-hoc `100 - trustScore` restatement would tell the user nothing
 * they could act on ahead of time.
 */
export function predictHallucinationRisk(
  field: string,
  value: string,
  retrieved: RetrievedCandidate[],
  quality: IndexQuality,
): HallucinationRisk {
  const reasons: string[] = [];
  let risk = 0;

  if (retrieved.length === 0) {
    risk += 55;
    reasons.push("No candidate evidence was retrieved anywhere in the document for this claim.");
  } else {
    const top = retrieved[0];
    if (top.valueSimilarity < 0.35) {
      risk += 30;
      reasons.push(`The closest text in the document only matches the claimed value at ${Math.round(top.valueSimilarity * 100)}%.`);
    } else if (top.valueSimilarity < 0.7) {
      risk += 15;
      reasons.push(`The closest match is partial (${Math.round(top.valueSimilarity * 100)}% similarity), so a near-miss correction is plausible.`);
    } else {
      reasons.push(`A strong textual match (${Math.round(top.valueSimilarity * 100)}%) was found before verification began.`);
    }

    if (top.block.legibility < 55) {
      risk += 22;
      reasons.push(`The matching region was read at only ${top.block.legibility}% legibility — small type, low contrast, or blur.`);
    }
    if (!top.block.boundingBox) {
      risk += 8;
      reasons.push("The matching region has no coordinates, so it cannot be grounded on the page.");
    }
    if (retrieved.length >= 4 && retrieved[1] && retrieved[1].score > top.score * 0.85) {
      risk += 12;
      reasons.push("Several regions match this claim about equally well, so the evidence is ambiguous.");
    }
  }

  if (quality.blockCount < 5) {
    risk += 15;
    reasons.push(`Only ${quality.blockCount} text block(s) were readable in this document.`);
  }
  if (quality.meanLegibility < 60) {
    risk += 12;
    reasons.push(`Overall document legibility is low (${quality.meanLegibility}%).`);
  }
  if (quality.smallTypeRatio > 0.5) {
    risk += 8;
    reasons.push(`${Math.round(quality.smallTypeRatio * 100)}% of located regions are very small on the page.`);
  }
  if (/\b\d/.test(value) && retrieved.every((candidate) => !candidate.strategies.includes("numeric"))) {
    risk += 14;
    reasons.push("The claim contains figures that were not found in any retrieved region.");
  }

  const score = Math.max(0, Math.min(100, risk));
  return { level: score >= 55 ? "HIGH" : score >= 28 ? "MEDIUM" : "LOW", score, reasons };
}

/* ── Claim relation graph ────────────────────────────────────────────────── */

export interface ClaimRelation {
  from: string;
  to: string;
  kind: "same-region" | "same-page" | "shared-evidence" | "lexical";
  strength: number;
}

/**
 * Derives relationships between claims from where their evidence actually sits, rather than from
 * a hardcoded document schema. Two claims proven by the same block are strongly related; two
 * proven by neighbouring blocks in the same region are weakly related.
 */
export function buildClaimGraph(
  claims: Array<{ id: string; field: string; value: string; blocks: TextBlock[] }>,
): ClaimRelation[] {
  const relations: ClaimRelation[] = [];

  for (let i = 0; i < claims.length; i++) {
    for (let j = i + 1; j < claims.length; j++) {
      const a = claims[i];
      const b = claims[j];
      const sharedBlocks = a.blocks.filter((block) => b.blocks.some((other) => other.id === block.id));

      if (sharedBlocks.length > 0) {
        relations.push({ from: a.id, to: b.id, kind: "shared-evidence", strength: 1 });
        continue;
      }
      if (a.blocks.length === 0 || b.blocks.length === 0) {
        const lexical = diceSimilarity(a.field, b.field);
        if (lexical >= 0.5) relations.push({ from: a.id, to: b.id, kind: "lexical", strength: round2(lexical) });
        continue;
      }

      const sameRegion = a.blocks.some((block) => b.blocks.some((other) => other.region === block.region && other.page === block.page));
      const samePage = a.blocks.some((block) => b.blocks.some((other) => other.page === block.page));
      if (sameRegion) relations.push({ from: a.id, to: b.id, kind: "same-region", strength: 0.6 });
      else if (samePage) relations.push({ from: a.id, to: b.id, kind: "same-page", strength: 0.3 });
    }
  }

  return relations;
}

const round2 = (value: number) => Math.round(value * 100) / 100;
