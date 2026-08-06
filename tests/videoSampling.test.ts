import { describe, it, expect } from "vitest";
import { frameCountFor, sampleTimestamps, timestampsFor } from "../src/lib/videoSampling.js";

/**
 * Frame counts were fixed literals (6 / 8) duplicated between the React page and the Python CLI,
 * and the sampling loop stepped i = 0…n-1 over `duration / n`, so the final portion of every video
 * was never looked at. These tests pin both the tier table and the end-to-end coverage.
 */

describe("adaptive frame count — summary", () => {
  it.each([
    [5, 4], [30, 4],          // <= 30s
    [30.1, 6], [45, 6], [60, 6],  // 30–60s
    [61, 8], [120, 8], [180, 8],  // 60s–3min
    [181, 12], [3600, 12],        // > 3min
  ])("%ss → %s frames", (duration, expected) => {
    expect(frameCountFor("summary", duration)).toBe(expected);
  });
});

describe("adaptive frame count — time capsule", () => {
  it.each([
    [5, 6], [30, 6],
    [30.1, 8], [60, 8],
    [61, 10], [180, 10],
    [181, 12], [7200, 12],
  ])("%ss → %s frames", (duration, expected) => {
    expect(frameCountFor("timecapsule", duration)).toBe(expected);
  });
});

describe("tier boundaries are inclusive at the top", () => {
  it("treats exactly 30s, 60s and 180s as the lower tier", () => {
    expect(frameCountFor("summary", 30)).toBe(4);
    expect(frameCountFor("summary", 60)).toBe(6);
    expect(frameCountFor("summary", 180)).toBe(8);
  });

  /* A stream or unloaded metadata reports 0/NaN/Infinity. Returning the shortest tier keeps the
     upload alive; throwing would lose it entirely. */
  it("falls back to the shortest tier for an unusable duration", () => {
    for (const bad of [0, -1, NaN, Infinity]) {
      expect(frameCountFor("summary", bad)).toBe(4);
      expect(frameCountFor("timecapsule", bad)).toBe(6);
    }
  });
});

describe("timestamps span the whole video", () => {
  it("starts at 0 and reaches the end — the bug this replaces", () => {
    const ts = sampleTimestamps(60, 6);
    expect(ts).toHaveLength(6);
    expect(ts[0]).toBe(0);
    // Old behaviour put the last sample at 5/6 × 60 = 50s, losing the final ten seconds.
    expect(ts[5]).toBeGreaterThan(59);
    expect(ts[5]).toBeLessThanOrEqual(60);
  });

  it("is evenly spaced", () => {
    const ts = sampleTimestamps(100, 5);
    const gaps = ts.slice(1).map((t, i) => t - ts[i]);
    // The final gap is fractionally shorter by the end-epsilon pullback.
    for (const gap of gaps) expect(gap).toBeCloseTo(25, 1);
  });

  it("never targets exactly the reported duration", () => {
    // Seeking to `duration` commonly decodes nothing, which would yield a blank final frame.
    const ts = sampleTimestamps(42, 8);
    expect(ts[ts.length - 1]).toBeLessThan(42);
  });

  it("stays inside the video for every sample", () => {
    for (const [d, n] of [[10, 4], [65, 8], [900, 12]] as const) {
      for (const t of sampleTimestamps(d, n)) {
        expect(t).toBeGreaterThanOrEqual(0);
        expect(t).toBeLessThanOrEqual(d);
      }
    }
  });

  it("handles degenerate counts without producing NaN", () => {
    expect(sampleTimestamps(60, 0)).toEqual([]);
    expect(sampleTimestamps(60, 1)).toEqual([0]);
    expect(sampleTimestamps(0, 4).every(Number.isFinite)).toBe(true);
  });
});

describe("timestampsFor ties the two together", () => {
  it("returns one timestamp per adaptive frame", () => {
    expect(timestampsFor("summary", 20)).toHaveLength(4);
    expect(timestampsFor("summary", 200)).toHaveLength(12);
    expect(timestampsFor("timecapsule", 20)).toHaveLength(6);
    expect(timestampsFor("timecapsule", 120)).toHaveLength(10);
  });

  /* Time capsule spends one Gemini call per frame, so the ceiling is a cost guarantee. */
  it("never exceeds 12 frames however long the video", () => {
    for (const d of [200, 3600, 86_400]) {
      expect(timestampsFor("summary", d).length).toBeLessThanOrEqual(12);
      expect(timestampsFor("timecapsule", d).length).toBeLessThanOrEqual(12);
    }
  });
});
