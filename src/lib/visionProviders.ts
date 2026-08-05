import { ProviderOption, VisionProviderId } from "@/types/truthlens";
import { invokeAi } from "@/integrations/aiClient";

/**
 * Provider preference.
 *
 * The list of providers and models is fetched from the backend registry — never hardcoded here.
 * A static list drifts the moment a vendor retires a model (Google retired the whole gemini-2.5-*
 * line for new projects), and it cannot know which keys this deployment actually has.
 *
 * The user's choice is stored locally and sent with each request; the server validates it against
 * what is really configured, so a stale preference degrades to the default rather than failing.
 */

const STORAGE_KEY = "truthlens.provider_preference";

export interface ProviderPreference {
  provider: VisionProviderId | null;
  model: string | null;
}

export function readPreference(): ProviderPreference {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { provider: null, model: null };
    const parsed = JSON.parse(raw) as ProviderPreference;
    return { provider: parsed.provider ?? null, model: parsed.model ?? null };
  } catch {
    return { provider: null, model: null };
  }
}

export function writePreference(preference: ProviderPreference): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preference));
  } catch {
    /* storage unavailable — requests simply use the server default */
  }
}

/** Merge the stored preference into a request body. Omitted keys let the server decide. */
export function withPreference<T extends Record<string, unknown>>(body: T): T & ProviderPreference {
  const { provider, model } = readPreference();
  return { ...body, provider, model };
}

let cache: ProviderOption[] | null = null;

/** Provider registry as the backend reports it. Cached for the session. */
export async function fetchProviders(force = false): Promise<ProviderOption[]> {
  if (cache && !force) return cache;
  const { data } = await invokeAi<{ providers?: ProviderOption[] }>("workspace", { action: "status" });
  cache = data?.providers ?? [];
  return cache;
}
