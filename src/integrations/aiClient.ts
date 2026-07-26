/**
 * Unified AI backend client.
 *
 * The AI backend is a set of serverless functions in /api that hold the
 * GEMINI_API_KEY server-side (never shipped to the browser) and proxy to
 * Google's Gemini API. Switch backends with the VITE_BACKEND env variable:
 *
 *   VITE_BACKEND=api     -> same-origin /api/* (Vercel functions; default)
 *   VITE_BACKEND=python  -> local FastAPI server at VITE_PYTHON_API
 *                           (default http://localhost:8000)
 */

type FnName = "generate-caption" | "refine-caption" | "analyze-video" | "verify-flag";

const backend = (import.meta.env.VITE_BACKEND ?? "api").toLowerCase();
const pythonBase = import.meta.env.VITE_PYTHON_API ?? "http://localhost:8000";

const baseUrl = backend === "python" ? pythonBase : "/api";

export async function invokeAi<T = any>(
  name: FnName,
  body: unknown
): Promise<{ data: T | null; error: Error | null }> {
  try {
    const res = await fetch(`${baseUrl}/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = "";
      try {
        const err = await res.json();
        detail = err.error ?? err.detail ?? "";
      } catch { /* non-JSON error body */ }
      throw new Error(`${name} failed (${res.status})${detail ? `: ${detail}` : ""}`);
    }
    const data = (await res.json()) as T;
    return { data, error: null };
  } catch (err) {
    return { data: null, error: err as Error };
  }
}

export const activeBackend = backend;
