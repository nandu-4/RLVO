import { supabase } from "@/integrations/supabase/client";

/**
 * Unified AI backend client.
 * Switches between Supabase Edge Functions and a local Python FastAPI server
 * based on the VITE_BACKEND environment variable.
 *
 *   VITE_BACKEND=supabase  -> calls Supabase Edge Functions (default)
 *   VITE_BACKEND=python    -> calls VITE_PYTHON_API (default http://localhost:8000)
 */

type FnName = "generate-caption" | "refine-caption" | "analyze-video";

const backend = (import.meta.env.VITE_BACKEND ?? "supabase").toLowerCase();
const pythonBase = import.meta.env.VITE_PYTHON_API ?? "http://localhost:8000";

async function callPython<T>(name: FnName, body: unknown): Promise<T> {
  const res = await fetch(`${pythonBase}/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Python backend ${name} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function invokeAi<T = any>(
  name: FnName,
  body: unknown
): Promise<{ data: T | null; error: Error | null }> {
  try {
    if (backend === "python") {
      const data = await callPython<T>(name, body);
      return { data, error: null };
    }
    const { data, error } = await supabase.functions.invoke(name, { body });
    return { data: data as T, error: error as Error | null };
  } catch (err) {
    return { data: null, error: err as Error };
  }
}

export const activeBackend = backend;
