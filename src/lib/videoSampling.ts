/**
 * Adaptive frame sampling for Video RLVO.
 *
 * WHAT CHANGED AND WHY
 * Frame counts were fixed literals — 6 for summary, 8 for time capsule — so a five-second clip and
 * a two-hour film were both reduced to the same handful of frames. Short clips paid for redundant
 * near-identical frames; long ones lost almost everything between samples. The count now follows
 * the video's duration.
 *
 * The tiers live in shared/video-sampling.json and are read by python/video_sampling.py as well, so
 * the two implementations cannot drift. They previously each carried their own copy of the numbers.
 *
 * THE END-OF-VIDEO FIX
 * The old loop spaced samples by `duration / n` and stepped `i = 0 … n-1`, so the last sample landed
 * at `(n-1)/n` of the way through and the closing stretch of every video was never looked at — for
 * summary, the final sixth. Spacing over `n - 1` intervals instead puts the first sample at 0 and
 * the last at the end, which is what "evenly spaced across the video" should have meant.
 */
import config from "../../shared/video-sampling.json";

export type SamplingMode = "summary" | "timecapsule";

interface Tier {
  maxSeconds: number | null;
  frames: number;
}

const TIERS = config.tiers as Record<SamplingMode, Tier[]>;
const END_EPSILON = config.endEpsilonSeconds as number;

/**
 * How many frames to take for a video of this length.
 *
 * A non-finite duration (a stream, or metadata that never loaded) falls back to the shortest tier
 * rather than throwing: fewer frames still produces a usable result, whereas a thrown error loses
 * the upload entirely.
 */
export function frameCountFor(mode: SamplingMode, durationSeconds: number): number {
  const tiers = TIERS[mode];
  if (!tiers?.length) throw new Error(`No sampling tiers configured for mode "${mode}".`);

  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return tiers[0].frames;

  const tier = tiers.find((t) => t.maxSeconds === null || durationSeconds <= t.maxSeconds);
  return (tier ?? tiers[tiers.length - 1]).frames;
}

/**
 * Evenly spaced timestamps covering the whole video, first at 0 and last at the end.
 *
 * Spacing divides by `count - 1`, not `count`, which is what makes the final sample reach the end.
 * The last target is pulled back by END_EPSILON because seeking to exactly `duration` commonly
 * decodes nothing.
 */
export function sampleTimestamps(durationSeconds: number, count: number): number[] {
  if (count <= 0) return [];
  const duration = Number.isFinite(durationSeconds) && durationSeconds > 0 ? durationSeconds : 0;
  const end = Math.max(0, duration - END_EPSILON);
  if (count === 1) return [0];

  return Array.from({ length: count }, (_, i) => Math.min((i * duration) / (count - 1), end));
}

/** Convenience: the timestamps this mode would use for a video of this length. */
export function timestampsFor(mode: SamplingMode, durationSeconds: number): number[] {
  return sampleTimestamps(durationSeconds, frameCountFor(mode, durationSeconds));
}
