/**
 * Best-effort in-memory rate limiter for the serverless handlers.
 *
 * HONEST LIMITATION: state lives in one function instance. Vercel runs many instances
 * concurrently and recycles them, so the effective ceiling is (limit x live instances),
 * not `limit`. This is a guardrail against a single client hammering one warm instance —
 * it is NOT a billing control. A durable limiter (Upstash/Redis keyed by identity) is
 * required before exposing this publicly; see RATE_LIMIT_BACKEND in .env.example.
 */

interface Window {
  count: number;
  resetAt: number;
}

const windows = new Map<string, Window>();
const MAX_TRACKED_KEYS = 5000;

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
}

export function rateLimit(key: string, limit: number, windowMs: number): RateLimitResult {
  const now = Date.now();
  const existing = windows.get(key);

  if (!existing || existing.resetAt <= now) {
    if (windows.size >= MAX_TRACKED_KEYS) evictExpired(now);
    windows.set(key, { count: 1, resetAt: now + windowMs });
    return { allowed: true, remaining: limit - 1, retryAfterSeconds: 0 };
  }

  existing.count += 1;
  if (existing.count > limit) {
    return { allowed: false, remaining: 0, retryAfterSeconds: Math.ceil((existing.resetAt - now) / 1000) };
  }
  return { allowed: true, remaining: limit - existing.count, retryAfterSeconds: 0 };
}

function evictExpired(now: number) {
  for (const [key, window] of windows) {
    if (window.resetAt <= now) windows.delete(key);
  }
  // Still full of live windows: drop the oldest so a flood of unique keys cannot grow the map.
  if (windows.size >= MAX_TRACKED_KEYS) {
    const oldest = [...windows.entries()].sort((a, b) => a[1].resetAt - b[1].resetAt).slice(0, MAX_TRACKED_KEYS / 2);
    for (const [key] of oldest) windows.delete(key);
  }
}

/** Stable-ish caller key: authenticated user when known, else the edge-reported client IP. */
export function callerKey(req: any, userId?: string): string {
  if (userId) return `user:${userId}`;
  const forwarded = String(req.headers?.["x-forwarded-for"] || "");
  const ip = forwarded.split(",")[0].trim() || req.socket?.remoteAddress || "unknown";
  return `ip:${ip}`;
}
