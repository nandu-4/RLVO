/**
 * Unified AI backend client.
 *
 * Routes to the configured backend (VITE_BACKEND: "python" or "api") with a fallback to the
 * other. Server-sent error messages are propagated verbatim — the previous version discarded
 * every non-2xx body and returned "Backend unavailable", which made a 413 (document too large),
 * a 429 (rate limited) and a 401 (sign-in required) all look like an outage.
 */

import { currentAccessToken } from "@/integrations/auth";

type FnName =
  | "generate-caption"
  | "refine-caption"
  | "analyze-video"
  | "verify-flag"
  | "verify-document"
  | "review-claim"
  | "analytics"
  | "batch-job"
  | "benchmark"
  | "workspace"
  | "extract-claims"
  | "review-queue";

const backend = (import.meta.env.VITE_BACKEND ?? "api").toLowerCase();
const pythonBase = import.meta.env.VITE_PYTHON_API ?? "http://localhost:8000";

const primaryUrl = backend === "python" ? pythonBase : "/api";
const secondaryUrl = backend === "python" ? "/api" : pythonBase;

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function attempt<T>(base: string, name: FnName, body: unknown): Promise<{ data: T } | { error: ApiError } | null> {
  // The verified Supabase session, or null for a guest. The API treats null as demo mode: it
  // still verifies documents, it simply stores nothing.
  const token = await currentAccessToken();
  let res: Response;
  try {
    res = await fetch(`${base}/${name}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });
  } catch {
    return null; // Transport failure — the caller may try the other backend.
  }

  if (res.ok) return { data: (await res.json()) as T };

  // The server responded, so this is a real answer, not an outage. Do not fall through.
  let message = `Request failed with status ${res.status}`;
  try {
    const payload = await res.json();
    if (payload?.error) message = String(payload.error);
  } catch {
    /* non-JSON error body: keep the status-based message */
  }
  return { error: new ApiError(message, res.status) };
}

export async function invokeAi<T = unknown>(name: FnName, body: unknown): Promise<{ data: T | null; error: ApiError | null }> {
  const primary = await attempt<T>(primaryUrl, name, body);
  if (primary && "data" in primary) return { data: primary.data, error: null };
  if (primary && "error" in primary) return { data: null, error: primary.error };

  if (primaryUrl !== secondaryUrl) {
    const secondary = await attempt<T>(secondaryUrl, name, body);
    if (secondary && "data" in secondary) return { data: secondary.data, error: null };
    if (secondary && "error" in secondary) return { data: null, error: secondary.error };
  }

  return { data: null, error: new ApiError(`Backend unreachable at ${primaryUrl}/${name}.`, 0) };
}

export const activeBackend = backend;
