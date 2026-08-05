/**
 * Bounding-box normalisation.
 *
 * The UI positions boxes with `left: ${x}%`, but nothing previously normalised what the provider
 * returned. Vision models emit boxes in at least three conventions — 0-1 fractions, 0-100
 * percentages, and Gemini's 0-1000 normalised [ymin, xmin, ymax, xmax] — so unnormalised boxes
 * landed off-canvas, mirrored, or at the wrong scale. Everything downstream now speaks one
 * convention: top-left origin, percentages of the page.
 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

type RawBox =
  | Partial<BoundingBox>
  | { ymin: number; xmin: number; ymax: number; xmax: number }
  | { top: number; left: number; bottom: number; right: number }
  | number[];

const clamp = (value: number, min = 0, max = 100) => Math.max(min, Math.min(max, value));
const isFinitePositive = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/**
 * Infer the coordinate space from the largest magnitude present, then rescale to percent.
 * A box whose values all sit in [0,1] is treated as fractional; up to 100 as percent; beyond
 * that as Gemini's 0-1000 space. Ambiguity is unavoidable (a genuinely tiny box in 0-1000 space
 * looks fractional), so we bias toward the interpretation that keeps the box on the page.
 *
 * Inferred PER AXIS, deliberately. Live output from gemini-2.5-flash-lite returns x/width/height
 * as correct percentages while y drifts far past 100 on the same box. Inferring one factor for
 * all four values let a single bad axis divide the whole box by ten — measured at 21 of 24 boxes
 * corrupted on a one-page invoice, which put every evidence overlay in the wrong place.
 */
function scaleFactor(values: number[]): number {
  const peak = Math.max(...values.map(Math.abs));
  if (peak <= 1.0001) return 100; // 0-1 fractional
  if (peak <= 100.0001) return 1; // already percent
  if (peak <= 1000.0001) return 100 / 1000; // Gemini 0-1000
  return 100 / peak; // unknown pixel space: normalise by the observed maximum
}

export function normalizeBoundingBox(raw: unknown): BoundingBox | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const box = raw as RawBox;

  // [ymin, xmin, ymax, xmax] — Gemini's array form.
  if (Array.isArray(box)) {
    if (box.length !== 4 || !box.every(isFinitePositive)) return undefined;
    const [ymin, xmin, ymax, xmax] = box;
    return fromCorners(xmin, ymin, xmax, ymax, scaleFactor(box));
  }

  const record = box as Record<string, unknown>;

  if (["ymin", "xmin", "ymax", "xmax"].every((key) => isFinitePositive(record[key]))) {
    const values = [record.ymin, record.xmin, record.ymax, record.xmax] as number[];
    return fromCorners(values[1], values[0], values[3], values[2], scaleFactor(values));
  }

  if (["top", "left", "bottom", "right"].every((key) => isFinitePositive(record[key]))) {
    const values = [record.top, record.left, record.bottom, record.right] as number[];
    return fromCorners(values[1], values[0], values[3], values[2], scaleFactor(values));
  }

  if (["x", "y", "width", "height"].every((key) => isFinitePositive(record[key]))) {
    const [rawX, rawY, rawW, rawH] = [record.x, record.y, record.width, record.height] as number[];

    /*
     * Measured reality: a single box can mix conventions — gemini-2.5-flash-lite returned
     * x/width/height as correct percentages while y alone drifted past 100. Neither one global
     * factor nor one factor per axis recovers that, because position and extent can disagree
     * too. So propose the sensible interpretations and take the first that lands on the page,
     * which is what "bias toward the interpretation that keeps the box on the page" means in
     * practice. Ordering matters: per-axis-pair first (the common case), then position-only
     * scaling (the mixed-convention case observed live).
     */
    const axisPair = { fx: scaleFactor([rawX, rawX + rawW]), fy: scaleFactor([rawY, rawY + rawH]) };
    const candidates: Array<{ fx: number; fy: number; fw: number; fh: number }> = [
      { fx: axisPair.fx, fy: axisPair.fy, fw: axisPair.fx, fh: axisPair.fy },
      // Position rescaled, extent left alone — recovers the mixed-convention box.
      { fx: scaleFactor([rawX]), fy: scaleFactor([rawY]), fw: scaleFactor([rawW]), fh: scaleFactor([rawH]) },
      // Everything already in percent.
      { fx: 1, fy: 1, fw: 1, fh: 1 },
    ];

    for (const f of candidates) {
      const width = clamp(rawW * f.fw, 0.1);
      const height = clamp(rawH * f.fh, 0.1);
      const x = clamp(rawX * f.fx, 0, 100 - Math.min(width, 100));
      const y = clamp(rawY * f.fy, 0, 100 - Math.min(height, 100));
      const box = round({ x, y, width: Math.min(width, 100 - x), height: Math.min(height, 100 - y) });
      if (plausible(box)) return box;
    }

    // Nothing placed it credibly. A wrong highlight is worse than none: it points a reviewer at
    // text that is not the evidence. Drop the overlay; the text and page number still stand.
    return undefined;
  }

  return undefined;
}

/**
 * Reject degenerate or off-page results rather than rendering a misleading highlight.
 * The height floor is deliberately low: a single 11pt line on a 792pt page is only ~1.4% tall,
 * so a stricter threshold would discard every legitimate text-line box.
 */
function plausible(box: BoundingBox): boolean {
  return (
    box.width >= 0.2 &&
    box.height >= 0.3 &&
    box.x >= 0 &&
    box.y >= 0 &&
    box.x + box.width <= 100.5 &&
    box.y + box.height <= 100.5
  );
}

function fromCorners(x1: number, y1: number, x2: number, y2: number, factor: number): BoundingBox | undefined {
  // Corner order is not guaranteed; sort rather than producing a negative extent.
  const left = clamp(Math.min(x1, x2) * factor);
  const top = clamp(Math.min(y1, y2) * factor);
  const right = clamp(Math.max(x1, x2) * factor);
  const bottom = clamp(Math.max(y1, y2) * factor);
  const width = right - left;
  const height = bottom - top;
  if (width <= 0 || height <= 0) return undefined;
  return round({ x: left, y: top, width, height });
}

const round = (box: BoundingBox): BoundingBox => ({
  x: Math.round(box.x * 100) / 100,
  y: Math.round(box.y * 100) / 100,
  width: Math.round(box.width * 100) / 100,
  height: Math.round(box.height * 100) / 100,
});

/** Intersection-over-union, used to measure layout agreement between evidence regions. */
export function iou(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1) return 0;
  const overlap = (x2 - x1) * (y2 - y1);
  const union = a.width * a.height + b.width * b.height - overlap;
  return union > 0 ? overlap / union : 0;
}

/** Vertical/horizontal proximity in percent, used to find neighbouring regions. */
export function distance(a: BoundingBox, b: BoundingBox): number {
  const ax = a.x + a.width / 2;
  const ay = a.y + a.height / 2;
  const bx = b.x + b.width / 2;
  const by = b.y + b.height / 2;
  return Math.hypot(ax - bx, ay - by);
}

/** Fraction of the page a box occupies — a proxy for how small the source text was. */
export const area = (box: BoundingBox): number => (box.width * box.height) / 100;
