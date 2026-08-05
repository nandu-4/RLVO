import { describe, it, expect } from "vitest";
import { normalizeBoundingBox, iou, distance, area } from "../api/_geometry.js";

/**
 * Regression suite for the coordinate bug that shipped broken.
 *
 * Live gemini-2.5-flash-lite output returned x/width/height as valid percentages while y drifted
 * past 100 on the same box. The original normalizer inferred ONE scale from all four values, so a
 * single bad axis divided the whole box by ten — 21 of 24 boxes corrupted, every evidence overlay
 * drawn in the wrong place. These cases are taken from that captured output.
 */
describe("normalizeBoundingBox", () => {
  it("passes through boxes already in percent", () => {
    expect(normalizeBoundingBox({ x: 11.77, y: 4.58, width: 49.01, height: 1.77 })).toEqual({
      x: 11.77, y: 4.58, width: 49.01, height: 1.77,
    });
  });

  it("does not corrupt a good x-axis when y is out of range (the shipped bug)", () => {
    // Real captured value: x/width/height correct, y wildly out of range.
    const box = normalizeBoundingBox({ x: 11.77, y: 124.95, width: 27.71, height: 1.15 });
    expect(box).toBeDefined();
    // The x axis must survive untouched — this is exactly what regressed.
    expect(box!.x).toBeCloseTo(11.77, 1);
    expect(box!.width).toBeCloseTo(27.71, 1);
  });

  it("keeps every axis on the page for the full captured sample", () => {
    const captured = [
      { x: 11.77, y: 4.58, width: 49.01, height: 1.77 },
      { x: 11.77, y: 17.7, width: 42.03, height: 1.15 },
      { x: 11.77, y: 94.95, width: 11.98, height: 2.13 },
      { x: 11.77, y: 124.95, width: 27.71, height: 1.15 },
      { x: 11.77, y: 205.45, width: 16.7, height: 1.15 },
      { x: 11.77, y: 389.8, width: 10.77, height: 1.15 },
    ];
    for (const raw of captured) {
      const box = normalizeBoundingBox(raw);
      if (!box) continue; // dropping is an acceptable outcome; drawing wrongly is not
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.y).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(100.5);
      expect(box.y + box.height).toBeLessThanOrEqual(100.5);
      expect(box.x).toBeCloseTo(raw.x, 0); // x never rescaled by a bad y
    }
  });

  it("reads Gemini's native [ymin, xmin, ymax, xmax] 0-1000 array", () => {
    const box = normalizeBoundingBox([18, 115, 42, 610]);
    expect(box).toEqual({ x: 11.5, y: 1.8, width: 49.5, height: 2.4 });
  });

  it("reads 0-1 fractional coordinates", () => {
    expect(normalizeBoundingBox({ x: 0.1, y: 0.2, width: 0.3, height: 0.05 })).toEqual({
      x: 10, y: 20, width: 30, height: 5,
    });
  });

  it("sorts reversed corners instead of producing negative extents", () => {
    const box = normalizeBoundingBox({ ymin: 400, xmin: 600, ymax: 100, xmax: 200 });
    expect(box!.width).toBeGreaterThan(0);
    expect(box!.height).toBeGreaterThan(0);
  });

  it("drops degenerate boxes rather than rendering a misleading highlight", () => {
    expect(normalizeBoundingBox({ x: 10, y: 10, width: 0, height: 0 })).toBeUndefined();
    expect(normalizeBoundingBox({ ymin: 5, xmin: 5, ymax: 5, xmax: 5 })).toBeUndefined();
  });

  it("rejects malformed input without throwing", () => {
    for (const bad of [null, undefined, {}, [1, 2], "nope", { x: "a", y: 1, width: 2, height: 3 }, [NaN, 1, 2, 3]]) {
      expect(normalizeBoundingBox(bad as unknown)).toBeUndefined();
    }
  });
});

describe("iou / distance / area", () => {
  it("scores identical boxes as 1 and disjoint boxes as 0", () => {
    const a = { x: 0, y: 0, width: 10, height: 10 };
    expect(iou(a, a)).toBe(1);
    expect(iou(a, { x: 50, y: 50, width: 10, height: 10 })).toBe(0);
  });

  it("scores partial overlap between 0 and 1", () => {
    const v = iou({ x: 0, y: 0, width: 10, height: 10 }, { x: 5, y: 0, width: 10, height: 10 });
    expect(v).toBeGreaterThan(0);
    expect(v).toBeLessThan(1);
  });

  it("measures centre distance and page footprint", () => {
    expect(distance({ x: 0, y: 0, width: 10, height: 10 }, { x: 0, y: 0, width: 10, height: 10 })).toBe(0);
    expect(area({ x: 0, y: 0, width: 10, height: 10 })).toBe(1);
  });
});
