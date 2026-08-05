import { useEffect, useState } from "react";
import { Cpu, ChevronDown, Check, AlertTriangle } from "lucide-react";
import { ProviderOption, VisionProviderId } from "@/types/truthlens";
import { fetchProviders, readPreference, writePreference } from "@/lib/visionProviders";

interface Props {
  onChange?: (preference: { provider: VisionProviderId | null; model: string | null }) => void;
}

/**
 * Provider + model picker, driven entirely by the backend registry.
 *
 * Nothing here is hardcoded. The old version listed five vendors as constants with four marked
 * "unavailable" forever; it could not know which keys a deployment actually holds, and it went
 * stale the moment Google retired the gemini-2.5-* line. Providers, their models and whether each
 * is usable now all come from the server.
 */
export default function VisionProviderSelector({ onChange }: Props) {
  const [providers, setProviders] = useState<ProviderOption[]>([]);
  const [open, setOpen] = useState(false);
  const [preference, setPreference] = useState(readPreference);

  useEffect(() => {
    void fetchProviders().then(setProviders);
  }, []);

  const configured = providers.filter((p) => p.configured);
  const active = providers.find((p) => p.id === preference.provider) ?? configured[0];
  const activeModel = preference.model ?? active?.defaultModel ?? "—";

  const select = (provider: ProviderOption, model: string) => {
    const next = { provider: provider.id, model };
    setPreference(next);
    writePreference(next);
    onChange?.(next);
    setOpen(false);
  };

  if (providers.length === 0) {
    return (
      <div className="glass-light rounded-xl px-3 py-2 text-xs text-muted-foreground flex items-center gap-2">
        <Cpu className="w-4 h-4" /> Loading providers…
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="listbox"
        className="glass-light rounded-xl px-3 py-2 flex items-center gap-2.5 border border-border hover:border-primary/50 transition-colors"
      >
        <Cpu className="w-4 h-4 text-primary shrink-0" />
        <span className="text-left">
          <span className="block text-xs font-semibold text-foreground leading-tight">{activeModel}</span>
          <span className="block text-[10px] text-muted-foreground leading-tight">{active?.vendor ?? "no provider configured"}</span>
        </span>
        <ChevronDown className={`w-3.5 h-3.5 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} aria-hidden="true" />
          <div role="listbox" className="absolute right-0 mt-2 w-80 glass-strong rounded-xl border border-border/60 shadow-2xl z-50 p-2 max-h-[26rem] overflow-y-auto">
            {providers.map((provider) => (
              <div key={provider.id} className="mb-2 last:mb-0">
                <div className="px-2 py-1.5 flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-foreground truncate">{provider.label}</div>
                    <div className="text-[10px] text-muted-foreground truncate">{provider.vendor}</div>
                  </div>
                  {!provider.configured && (
                    <span className="text-[9px] font-bold uppercase text-warning flex items-center gap-1 shrink-0" title={`Set ${provider.keyVar}`}>
                      <AlertTriangle className="w-3 h-3" /> no key
                    </span>
                  )}
                </div>

                {provider.models.map((model) => {
                  const isActive = active?.id === provider.id && activeModel === model;
                  return (
                    <button
                      key={model}
                      role="option"
                      aria-selected={isActive}
                      disabled={!provider.configured}
                      onClick={() => select(provider, model)}
                      className={`w-full text-left px-2.5 py-1.5 rounded-lg text-[11px] font-mono flex items-center justify-between gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                        isActive ? "bg-primary/20 text-primary" : "text-muted-foreground hover:bg-surface-light hover:text-foreground"
                      }`}
                    >
                      <span className="truncate">{model}</span>
                      {isActive && <Check className="w-3.5 h-3.5 shrink-0" />}
                    </button>
                  );
                })}
              </div>
            ))}
            <p className="text-[10px] text-muted-foreground px-2 pt-2 border-t border-border/40 leading-relaxed">
              If the chosen provider fails, TruthLens automatically retries with another configured provider and tells
              you which one produced the result.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
