import { activeModel as geminiDefaultModel } from "../_gemini.js";
import { createGeminiAdapter } from "./gemini.js";
import { createOpenRouterAdapter, DEFAULT_OPENROUTER_MODEL, OPENROUTER_MODELS } from "./openrouter.js";
import { createHuggingFaceAdapter, DEFAULT_HUGGINGFACE_MODEL } from "./huggingface.js";
import type { VisionProviderAdapter } from "./types.js";

/**
 * Provider registry.
 *
 * A provider is a *vendor gateway* (Gemini, OpenRouter); a model is chosen within it. Adding a
 * vendor means one new file implementing VisionProviderAdapter plus one entry here — the
 * pipeline, retrieval engine, scoring, persistence and UI are untouched.
 */

export type ProviderId = "gemini" | "openrouter" | "huggingface";

interface ProviderDefinition {
  id: ProviderId;
  label: string;
  vendor: string;
  /** Env var that switches this provider on. */
  keyVar: string;
  models: string[];
  defaultModel: string;
  create(model?: string): VisionProviderAdapter;
  isConfigured(): boolean;
}

const DEFINITIONS: ProviderDefinition[] = [
  {
    id: "gemini",
    label: "Google Gemini (direct)",
    vendor: "Google DeepMind",
    keyVar: "GEMINI_API_KEY",
    // Aliases, not pinned versions: Google retires pinned names for new projects, which 404s.
    models: ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest"],
    defaultModel: geminiDefaultModel,
    create: (model) => createGeminiAdapter(model),
    isConfigured: () => Boolean(process.env.GEMINI_API_KEY),
  },
  {
    id: "openrouter",
    label: "OpenRouter (multi-vendor gateway)",
    vendor: "Anthropic · OpenAI · Google · Alibaba",
    keyVar: "OPENROUTER_API_KEY",
    models: [...OPENROUTER_MODELS],
    defaultModel: DEFAULT_OPENROUTER_MODEL,
    create: (model) => createOpenRouterAdapter(model),
    isConfigured: () => Boolean(process.env.OPENROUTER_API_KEY),
  },
  { id: "huggingface", label: "Hugging Face", vendor: "Hugging Face", keyVar: "HUGGINGFACE_API_KEY", models: [DEFAULT_HUGGINGFACE_MODEL], defaultModel: DEFAULT_HUGGINGFACE_MODEL, create: (model) => createHuggingFaceAdapter(model), isConfigured: () => Boolean(process.env.HUGGINGFACE_API_KEY) },
];

const byId = (id: string) => DEFINITIONS.find((d) => d.id === id);

/** Deployment-wide default, overridable per request. */
export function defaultProviderId(): ProviderId {
  const configured = process.env.DEFAULT_PROVIDER as ProviderId | undefined;
  if (configured && byId(configured)?.isConfigured()) return configured;
  return (DEFINITIONS.find((d) => d.isConfigured())?.id ?? "gemini") as ProviderId;
}

export interface ResolvedProvider {
  adapter: VisionProviderAdapter;
  providerId: ProviderId;
  providerLabel: string;
  model: string;
}

/**
 * Resolve a provider + model, validating the model belongs to that provider.
 *
 * A client may request anything; an unknown model is replaced by the provider's default rather
 * than passed through to the vendor, so a typo cannot turn into an opaque upstream 404.
 */
export function resolveProvider(providerId?: string, model?: string): ResolvedProvider | undefined {
  const definition = byId(providerId ?? defaultProviderId()) ?? byId(defaultProviderId());
  if (!definition?.isConfigured()) return undefined;
  const chosen = model && definition.models.includes(model) ? model : definition.defaultModel;
  return {
    adapter: definition.create(chosen),
    providerId: definition.id,
    providerLabel: definition.label,
    model: chosen,
  };
}

/**
 * Resolution order for automatic failover.
 *
 * The chain walks MODELS, not just providers. That distinction is the whole point:
 *
 * MEASURED against a live free-tier key — Gemini's daily allowance is billed per model, with
 * quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. With `gemini-flash-latest`
 * exhausted (429), `gemini-flash-lite-latest` answered normally on the same key at the same
 * moment. A chain of one-model-per-provider abandoned Gemini on that first 429 and fell through
 * to vendors that were genuinely out of credit, so the whole request failed while a working
 * model sat unused. Sibling models are tried before giving up on a vendor.
 *
 * Order within a provider is default-model first, then its siblings; providers keep registry
 * order, with any explicitly requested provider promoted to the front. Exhausted-quota and
 * insufficient-credit responses come back in well under a second, so the extra candidates cost
 * negligible wall-clock — and the caller's own time guard drops the tail if the budget runs low.
 */
export function resolutionChain(providerId?: string, model?: string): ResolvedProvider[] {
  const requested = providerId ? byId(providerId) : undefined;
  const ordered = requested ? [requested, ...DEFINITIONS.filter((d) => d.id !== requested.id)] : DEFINITIONS;

  const chain: ResolvedProvider[] = [];
  const seen = new Set<string>();

  for (const definition of ordered) {
    if (!definition.isConfigured()) continue;
    // An explicitly requested model leads, but only for the provider it belongs to.
    const preferred = definition === requested && model && definition.models.includes(model) ? [model] : [];
    for (const candidate of [...preferred, definition.defaultModel, ...definition.models]) {
      const resolved = resolveProvider(definition.id, candidate);
      if (!resolved) continue;
      const key = `${resolved.providerId}:${resolved.model}`;
      if (seen.has(key)) continue;
      seen.add(key);
      chain.push(resolved);
    }
  }
  return chain;
}

/** Registry view for the admin surface and the provider picker. */
export function providerStatus() {
  return DEFINITIONS.map((d) => ({
    id: d.id,
    label: d.label,
    vendor: d.vendor,
    keyVar: d.keyVar,
    configured: d.isConfigured(),
    models: d.models,
    defaultModel: d.defaultModel,
  }));
}

/**
 * Models the benchmark can run, across every configured provider.
 *
 * gemini-2.0-* and gemini-2.5-* are deliberately absent: measured against live keys, 2.0 returns
 * "limit: 0" and 2.5 returns 404 "no longer available to new users" on newly created projects.
 */
export function benchmarkTargets(): Array<{
  id: string;
  label: string;
  vendor: string;
  available: boolean;
  reason?: string;
  adapter?: VisionProviderAdapter;
}> {
  const override = process.env.BENCHMARK_MODELS?.split(",").map((m) => m.trim()).filter(Boolean);

  return DEFINITIONS.flatMap((definition) => {
    const models = override?.filter((m) => definition.models.includes(m)) ?? definition.models;
    return models.map((model) => ({
      id: `${definition.id}:${model}`,
      label: model,
      vendor: definition.vendor,
      available: definition.isConfigured(),
      reason: definition.isConfigured() ? undefined : `${definition.keyVar} is not configured.`,
      adapter: definition.isConfigured() ? definition.create(model) : undefined,
    }));
  });
}

export type { VisionProviderAdapter } from "./types.js";
