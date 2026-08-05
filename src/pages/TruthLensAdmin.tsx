import { useEffect, useState } from "react";
import { Key, Brain, Shield, Settings, Database, CheckCircle2, MinusCircle, AlertTriangle, Loader2, Copy, Trash2, Download, RefreshCw, Activity } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import { invokeAi } from "@/integrations/aiClient";
import { forgetWorkspace, replaceWorkspaceToken, workspaceToken } from "@/lib/workspace";
import { readPreference, writePreference } from "@/lib/visionProviders";
import { cn } from "@/lib/utils";

type Tab = "workspace" | "models" | "rules" | "compliance" | "activity";

interface Control {
  id: string;
  label: string;
  state: "enforced" | "partial" | "absent";
  detail: string;
}

interface Status {
  available: boolean;
  reason?: string;
  activeModel?: string;
  workspace?: { id: string; name: string; retentionDays: number; documentCount: number; oldestDocumentAt: string | null; expiredAwaitingPurge: number };
  providers?: Array<{ id: string; label: string; vendor: string; keyVar: string; configured: boolean; models: string[]; defaultModel: string }>;
  benchmarkTargets?: Array<{ id: string; label: string; vendor: string; available: boolean; reason?: string }>;
  compliance?: { controls: Control[]; caveat: string };
}

interface ActivityLog {
  apiActivity: Array<{ id: string; route: string; action: string; statusCode: number; durationMs: number | null; createdAt: string }>;
  reviewDecisions: Array<{ id: string; decision: string; reviewerName: string; reviewerNotes: string | null; documentName: string; createdAt: string }>;
}

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "workspace", label: "Workspace", icon: <Key className="w-4 h-4" /> },
  { id: "models", label: "Models", icon: <Brain className="w-4 h-4" /> },
  { id: "rules", label: "Rules", icon: <Shield className="w-4 h-4" /> },
  { id: "compliance", label: "Compliance", icon: <Settings className="w-4 h-4" /> },
  { id: "activity", label: "Activity", icon: <Activity className="w-4 h-4" /> },
];

const STATE_STYLE = {
  enforced: { icon: <CheckCircle2 className="w-3.5 h-3.5" />, cls: "text-success", label: "Enforced" },
  partial: { icon: <AlertTriangle className="w-3.5 h-3.5" />, cls: "text-warning", label: "Partial" },
  absent: { icon: <MinusCircle className="w-3.5 h-3.5" />, cls: "text-muted-foreground", label: "Not provided" },
} as const;

export default function TruthLensAdmin() {
  const [activeTab, setActiveTab] = useState<Tab>("workspace");
  const [status, setStatus] = useState<Status | null>(null);
  const [activity, setActivity] = useState<ActivityLog | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [retention, setRetention] = useState(30);
  const [tokenVisible, setTokenVisible] = useState(false);
  const [preference, setPreference] = useState(readPreference);
  const [importValue, setImportValue] = useState("");

  const load = async () => {
    setLoading(true);
    const { data } = await invokeAi<Status>("workspace", { action: "status" });
    if (data) {
      setStatus(data);
      setName(data.workspace?.name ?? "");
      setRetention(data.workspace?.retentionDays ?? 30);
    }
    setLoading(false);
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (activeTab === "activity" && !activity) {
      void invokeAi<ActivityLog>("workspace", { action: "activity" }).then(({ data }) => data && setActivity(data));
    }
  }, [activeTab, activity]);

  const saveSettings = async () => {
    const { data, error } = await invokeAi<{ workspace: { name: string; retentionDays: number } }>("workspace", {
      action: "settings",
      name,
      retentionDays: retention,
    });
    setNotice(error ? error.message : `Saved. Retention now applies to all ${status?.workspace?.documentCount ?? 0} stored document(s).`);
    if (data) void load();
  };

  const runAction = async (action: "purge" | "erase", confirmText: string) => {
    if (!window.confirm(confirmText)) return;
    const { data, error } = await invokeAi<{ purged?: number; erased?: boolean }>("workspace", { action });
    setNotice(error ? error.message : action === "purge" ? `Purged ${data?.purged ?? 0} expired document(s).` : "All workspace data erased.");
    setActivity(null);
    void load();
  };

  const token = workspaceToken();

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-12 px-4 md:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8 flex items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">
                Admin <span className="gradient-text">Panel</span>
              </h1>
              <p className="text-xs md:text-sm text-muted-foreground mt-1">
                Workspace, model backends, verification rules and compliance posture for this deployment.
              </p>
            </div>
            <button onClick={load} disabled={loading} className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 disabled:opacity-50">
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Refresh
            </button>
          </div>

          {notice && <GlassCard hover={false} className="p-3.5 mb-5"><p className="text-xs text-foreground">{notice}</p></GlassCard>}

          <div className="flex flex-col lg:flex-row gap-6">
            <div className="lg:w-56 flex-shrink-0">
              <GlassCard hover={false} className="p-2">
                <div className="flex lg:flex-col gap-1 overflow-x-auto">
                  {TABS.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={cn(
                        "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap",
                        activeTab === tab.id
                          ? "bg-primary/15 text-white border border-primary/30"
                          : "text-muted-foreground hover:text-foreground hover:bg-surface-light",
                      )}
                    >
                      {tab.icon}
                      {tab.label}
                    </button>
                  ))}
                </div>
              </GlassCard>
            </div>

            <div className="flex-1 space-y-6">
              {/* ── Workspace ── */}
              {activeTab === "workspace" && (
                <>
                  {!status?.available ? (
                    <GlassCard hover={false} className="p-10 text-center">
                      <Database className="w-9 h-9 text-primary mx-auto mb-4" />
                      <h3 className="text-base font-bold">Workspace storage is not configured</h3>
                      <p className="text-sm text-muted-foreground mt-2 leading-relaxed max-w-lg mx-auto">{status?.reason}</p>
                    </GlassCard>
                  ) : (
                    <>
                      <GlassCard hover={false}>
                        <h3 className="text-sm font-semibold mb-1">Workspace settings</h3>
                        <p className="text-[11px] text-muted-foreground mb-4">
                          {status.workspace?.documentCount} document(s) stored
                          {status.workspace?.oldestDocumentAt && ` · oldest ${new Date(status.workspace.oldestDocumentAt).toLocaleDateString()}`}
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                          <div>
                            <label className="block text-muted-foreground mb-1.5">Workspace name</label>
                            <input
                              value={name}
                              onChange={(event) => setName(event.target.value)}
                              className="glass-light p-2.5 rounded-lg w-full bg-transparent border border-border text-foreground focus:outline-none focus:border-primary"
                            />
                          </div>
                          <div>
                            <label className="block text-muted-foreground mb-1.5">Data retention (days)</label>
                            <input
                              type="number"
                              min={1}
                              max={3650}
                              value={retention}
                              onChange={(event) => setRetention(Number(event.target.value))}
                              className="glass-light p-2.5 rounded-lg w-full bg-transparent border border-border text-foreground focus:outline-none focus:border-primary"
                            />
                          </div>
                        </div>
                        <button onClick={saveSettings} className="btn-primary text-xs py-2 px-4 mt-4 relative z-10">Save settings</button>
                      </GlassCard>

                      <GlassCard hover={false}>
                        <h3 className="text-sm font-semibold mb-1">Workspace key</h3>
                        {/* The trade-off is stated here, not buried in docs — the user is the only one who can hold this. */}
                        <p className="text-[11px] text-muted-foreground mb-3 leading-relaxed">
                          TruthLens has no accounts. This key <em>is</em> your workspace: it is stored only in this
                          browser, and the server keeps only its hash. Anyone holding it can read, review and erase this
                          workspace, and if you lose it nothing can recover the data — we hold no identifier that could
                          prove it was yours. Save it somewhere safe if this work matters, and do not use this mode for
                          regulated personal or health data.
                        </p>
                        <div className="glass-light rounded-lg p-3 flex items-center gap-2 mb-3">
                          <code className="text-[11px] font-mono flex-1 truncate">
                            {tokenVisible ? token : "•".repeat(43)}
                          </code>
                          <button onClick={() => setTokenVisible((v) => !v)} className="text-[10px] text-primary font-semibold shrink-0">
                            {tokenVisible ? "Hide" : "Reveal"}
                          </button>
                          <button
                            onClick={() => {
                              if (token) void navigator.clipboard.writeText(token).then(() => setNotice("Workspace key copied to clipboard."));
                            }}
                            className="p-1.5 rounded hover:bg-surface-light text-muted-foreground shrink-0"
                            title="Copy key"
                          >
                            <Copy className="w-3.5 h-3.5" />
                          </button>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          <input
                            value={importValue}
                            onChange={(event) => setImportValue(event.target.value)}
                            placeholder="Paste a key to switch workspace…"
                            className="glass-light px-3 py-2 rounded-lg text-xs border border-border bg-transparent flex-1 min-w-[220px] focus:outline-none focus:border-primary"
                          />
                          <button
                            onClick={() => {
                              if (replaceWorkspaceToken(importValue)) window.location.reload();
                              else setNotice("That does not look like a valid workspace key.");
                            }}
                            className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5"
                          >
                            <Download className="w-3.5 h-3.5" /> Switch
                          </button>
                        </div>
                      </GlassCard>

                      <GlassCard hover={false}>
                        <h3 className="text-sm font-semibold mb-1">Data controls</h3>
                        <p className="text-[11px] text-muted-foreground mb-4">
                          {status.workspace?.expiredAwaitingPurge ?? 0} document(s) are past their retention window and awaiting purge.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={() => runAction("purge", "Delete all documents past their retention window? This cannot be undone.")}
                            className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5"
                          >
                            <Trash2 className="w-3.5 h-3.5 text-warning" /> Apply retention now
                          </button>
                          <button
                            onClick={() => runAction("erase", "Permanently delete EVERY document, claim, evidence record and audit entry in this workspace? This cannot be undone.")}
                            className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5"
                          >
                            <Trash2 className="w-3.5 h-3.5 text-danger" /> Erase all workspace data
                          </button>
                          <button
                            onClick={() => {
                              if (window.confirm("Forget this workspace key in this browser? The stored data stays on the server but becomes unreachable without the key.")) {
                                forgetWorkspace();
                                window.location.reload();
                              }
                            }}
                            className="btn-secondary text-xs py-2 px-3.5"
                          >
                            Forget key on this device
                          </button>
                        </div>
                      </GlassCard>
                    </>
                  )}
                </>
              )}

              {/* ── Models ── */}
              {activeTab === "models" && (
                <>
                  <GlassCard hover={false}>
                    <h3 className="text-sm font-semibold mb-1">Active provider and model</h3>
                    <p className="text-[11px] text-muted-foreground mb-4">
                      Choose which vendor gateway and model TruthLens uses. The selection is stored in this browser and
                      sent with each request; the server validates it against what is actually configured, so no code
                      change or redeploy is needed. Currently running:{" "}
                      <span className="font-mono text-foreground">{status?.activeModel ?? "—"}</span>
                    </p>

                    <div className="space-y-3">
                      {(status?.providers ?? []).map((provider) => (
                        <div key={provider.id} className="glass-light rounded-lg p-3">
                          <div className="flex items-center justify-between gap-3 mb-2">
                            <div className="min-w-0">
                              <div className="text-sm font-semibold text-foreground">{provider.label}</div>
                              <div className="text-[11px] text-muted-foreground">{provider.vendor}</div>
                            </div>
                            <span
                              className={cn(
                                "flex items-center gap-1.5 text-[11px] font-medium shrink-0",
                                provider.configured ? "text-success" : "text-warning",
                              )}
                            >
                              {provider.configured ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                              {provider.configured ? "Key configured" : `Set ${provider.keyVar}`}
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-1.5">
                            {provider.models.map((model) => {
                              const selected =
                                (preference.provider ?? status?.providers?.find((p) => p.configured)?.id) === provider.id &&
                                (preference.model ?? provider.defaultModel) === model;
                              return (
                                <button
                                  key={model}
                                  disabled={!provider.configured}
                                  onClick={() => {
                                    const next = { provider: provider.id as never, model };
                                    setPreference(next);
                                    writePreference(next);
                                    setNotice(`Now using ${model} via ${provider.label}. Applies to the next verification.`);
                                  }}
                                  className={cn(
                                    "px-2.5 py-1.5 rounded-lg text-[11px] font-mono border transition-all disabled:opacity-40 disabled:cursor-not-allowed",
                                    selected
                                      ? "bg-primary/20 border-primary text-primary font-bold"
                                      : "glass-light border-border text-muted-foreground hover:text-foreground",
                                  )}
                                >
                                  {model}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>

                    <button
                      onClick={() => {
                        const next = { provider: null, model: null };
                        setPreference(next);
                        writePreference(next);
                        setNotice("Cleared. Requests will use the deployment default.");
                      }}
                      className="btn-secondary text-xs py-2 px-3.5 mt-4"
                    >
                      Use deployment default
                    </button>
                  </GlassCard>

                  <GlassCard hover={false}>
                    <h3 className="text-sm font-semibold mb-1">Automatic failover</h3>
                    <p className="text-[11px] text-muted-foreground mb-3">
                      If the selected provider fails — quota exhausted, vendor outage, retired model — TruthLens retries
                      with the next configured provider inside the same request, and the result reports which one
                      actually produced it. Order is: your selection first, then every other configured provider.
                    </p>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {(status?.providers ?? []).filter((p) => p.configured).map((p, i) => (
                        <span key={p.id} className="flex items-center gap-2">
                          {i > 0 && <span className="text-muted-foreground">→</span>}
                          <span className="glass-light px-2.5 py-1 rounded-lg text-foreground">{p.label}</span>
                        </span>
                      ))}
                      {(status?.providers ?? []).filter((p) => p.configured).length < 2 && (
                        <span className="text-muted-foreground">
                          Only one provider configured — add a second key to enable failover.
                        </span>
                      )}
                    </div>
                  </GlassCard>

                  <GlassCard hover={false}>
                    <h3 className="text-sm font-semibold mb-1">Benchmark targets</h3>
                    <p className="text-[11px] text-muted-foreground mb-4">Models the benchmark page can run, across every configured provider.</p>
                    <div className="space-y-2 text-xs">
                      {(status?.benchmarkTargets ?? []).map((target) => (
                        <div key={target.id} className="glass-light p-3 rounded-lg flex justify-between items-center gap-3">
                          <div className="min-w-0">
                            <div className="font-mono font-semibold truncate">{target.label}</div>
                            <div className="text-muted-foreground">{target.vendor}</div>
                          </div>
                          <span className={cn("shrink-0 text-right", target.available ? "text-success" : "text-muted-foreground")}>
                            {target.available ? "Available" : target.reason}
                          </span>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </>
              )}

              {/* ── Rules ── */}
              {activeTab === "rules" && (
                <GlassCard hover={false}>
                  <h3 className="text-sm font-semibold mb-1">Verification rules in force</h3>
                  <p className="text-xs text-muted-foreground mb-4">
                    These are the decision rules the server actually enforces in code. They are not yet editable —
                    changing them requires a deploy.
                  </p>
                  <div className="space-y-2 text-xs">
                    {[
                      { name: "Claim-blind transcription", value: "Enforced", detail: "The document is transcribed before any claim is shown to the model, so evidence cannot be shaped to fit the answer being checked." },
                      { name: "Citation allowlist", value: "Enforced", detail: "The verifier may only cite evidence ids the retrieval engine returned. Unrecognised citations are discarded." },
                      { name: "Evidence required to auto-decide", value: "Enforced", detail: "A verified or corrected status with no resolvable cited evidence is downgraded to needs_review." },
                      { name: "Grounded corrections only", value: "Enforced", detail: "A corrected value is dropped unless it appears verbatim in cited evidence." },
                      { name: "Container metadata extraction", value: "Blocked", detail: "PDF internals, xref, TeX producer strings and binary stream text are rejected at index time." },
                      { name: "Unmeasured signal handling", value: "Excluded", detail: "A trust signal nothing measured is excluded from the weighted mean rather than defaulted to another signal's value." },
                      { name: "Document risk thresholds", value: "85 / 60", detail: "Document risk is LOW at or above 85 with no open reviews, MEDIUM at or above 60, otherwise HIGH RISK." },
                      { name: "Pre-verification risk thresholds", value: "55 / 28", detail: "Predicted from retrieval strength and page legibility before verification; HIGH at or above 55, MEDIUM at or above 28." },
                    ].map((rule) => (
                      <div key={rule.name} className="glass-light p-3 rounded-lg">
                        <div className="flex justify-between items-center gap-3 mb-1">
                          <span className="font-semibold text-foreground">{rule.name}</span>
                          <span className="font-mono text-primary shrink-0">{rule.value}</span>
                        </div>
                        <p className="text-muted-foreground leading-relaxed">{rule.detail}</p>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              )}

              {/* ── Compliance ── */}
              {activeTab === "compliance" && (
                <GlassCard hover={false}>
                  <h3 className="text-sm font-semibold mb-1">Compliance posture</h3>
                  <p className="text-[11px] text-muted-foreground mb-4 leading-relaxed">{status?.compliance?.caveat}</p>
                  <div className="space-y-2 text-xs">
                    {(status?.compliance?.controls ?? []).map((control) => {
                      const style = STATE_STYLE[control.state];
                      return (
                        <div key={control.id} className="glass-light p-3 rounded-lg">
                          <div className="flex justify-between items-center gap-3 mb-1">
                            <span className="font-semibold text-foreground">{control.label}</span>
                            <span className={cn("flex items-center gap-1.5 font-bold uppercase text-[10px] tracking-wider shrink-0", style.cls)}>
                              {style.icon} {style.label}
                            </span>
                          </div>
                          <p className="text-muted-foreground leading-relaxed">{control.detail}</p>
                        </div>
                      );
                    })}
                  </div>
                </GlassCard>
              )}

              {/* ── Activity ── */}
              {activeTab === "activity" && (
                <>
                  <GlassCard hover={false}>
                    <h3 className="text-sm font-semibold mb-1">Human review decisions</h3>
                    <p className="text-[11px] text-muted-foreground mb-4">
                      Reviewer names are self-declared — with no accounts, this records who <em>said</em> they made a
                      decision, not a verified identity.
                    </p>
                    <div className="space-y-2 text-xs max-h-80 overflow-y-auto">
                      {(activity?.reviewDecisions ?? []).length === 0 ? (
                        <p className="text-muted-foreground py-4 text-center">No human decisions recorded yet.</p>
                      ) : (
                        activity?.reviewDecisions.map((entry) => (
                          <div key={entry.id} className="glass-light p-3 rounded-lg">
                            <div className="flex justify-between items-center gap-3">
                              <span className="font-semibold text-foreground capitalize">{entry.decision}</span>
                              <span className="text-muted-foreground shrink-0">{new Date(entry.createdAt).toLocaleString()}</span>
                            </div>
                            <p className="text-muted-foreground mt-0.5">
                              {entry.reviewerName} · {entry.documentName}
                            </p>
                            {entry.reviewerNotes && <p className="text-muted-foreground mt-1 italic">{entry.reviewerNotes}</p>}
                          </div>
                        ))
                      )}
                    </div>
                  </GlassCard>

                  <GlassCard hover={false}>
                    <h3 className="text-sm font-semibold mb-4">API activity log</h3>
                    <div className="space-y-1.5 text-xs max-h-96 overflow-y-auto">
                      {(activity?.apiActivity ?? []).length === 0 ? (
                        <p className="text-muted-foreground py-4 text-center">No API activity recorded yet.</p>
                      ) : (
                        activity?.apiActivity.map((entry) => (
                          <div key={entry.id} className="glass-light p-2.5 rounded-lg flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <span className="font-mono text-primary text-[10px]">{entry.route}</span>
                              <span className="text-foreground ml-2">{entry.action}</span>
                            </div>
                            <div className="flex items-center gap-3 shrink-0 text-muted-foreground">
                              {entry.durationMs !== null && <span className="tabular-nums">{entry.durationMs} ms</span>}
                              <span className={entry.statusCode < 400 ? "text-success" : "text-danger"}>{entry.statusCode}</span>
                              <span>{new Date(entry.createdAt).toLocaleTimeString()}</span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </GlassCard>
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
