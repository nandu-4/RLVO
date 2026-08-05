import { useCallback, useEffect, useState } from "react";
import { Scale, Loader2, Upload, X, FileText, AlertTriangle, Info, Trophy } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import { invokeAi } from "@/integrations/aiClient";

interface Target {
  id: string;
  label: string;
  vendor: string;
  available: boolean;
  reason?: string;
}

interface ModelResult {
  id: string;
  label: string;
  available: boolean;
  skipped?: boolean;
  error?: string;
  documentType?: string;
  claimsGenerated?: number;
  verified?: number;
  corrections?: number;
  unsupported?: number;
  needsReview?: number;
  trustScore?: number;
  hallucinationsCaught?: number;
  decisionRate?: number;
  evidenceCitationRate?: number;
  signalsMeasuredAvg?: number;
  blocksTranscribed?: number;
  meanLegibility?: number;
  transcribeMs?: number;
  verifyMs?: number;
  totalMs?: number;
  claims?: Array<{ field: string; status: string; trustScore: number; verifiedValue: string | null }>;
}

interface BenchmarkRun {
  runId: string;
  fileName: string;
  claimCount: number;
  results: ModelResult[];
  totalMs: number;
  disclaimer: string;
}

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

export default function TruthLensBenchmark() {
  const [targets, setTargets] = useState<Target[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [claimsText, setClaimsText] = useState("");
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void invokeAi<{ benchmarkTargets?: Target[] }>("workspace", { action: "status" }).then(({ data }) => {
      const list = data?.benchmarkTargets ?? [];
      setTargets(list);
      setSelected(list.filter((target) => target.available).slice(0, 3).map((target) => target.id));
    });
  }, []);

  const handleFile = useCallback((f: File) => {
    if (f.size > MAX_UPLOAD_BYTES) {
      setError(`${(f.size / 1024 / 1024).toFixed(1)} MB exceeds the 4 MB limit.`);
      return;
    }
    setError(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = (event) => setPreview(event.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const start = async () => {
    if (!file || !preview) return;
    const upstreamClaims = claimsText
      .split("\n")
      .map((line) => line.split(/:(.+)/))
      .filter(([field, value]) => field?.trim() && value?.trim())
      .map(([field, value]) => ({ field: field.trim(), value: value.trim() }));

    if (upstreamClaims.length === 0) {
      setError("Paste the claims to benchmark, one per line as Field: Value.");
      return;
    }
    if (selected.length === 0) {
      setError("Select at least one model to benchmark.");
      return;
    }

    setRunning(true);
    setError(null);
    const { data, error } = await invokeAi<BenchmarkRun>("benchmark", {
      image: preview,
      fileName: file.name,
      upstreamClaims,
      models: selected,
    });
    if (error) setError(error.message);
    else setRun(data);
    setRunning(false);
  };

  const completed = (run?.results || []).filter((result) => !result.error && !result.skipped && result.available);
  const bestTrust = completed.length > 0 ? Math.max(...completed.map((result) => result.trustScore ?? 0)) : null;
  const fastest = completed.length > 0 ? Math.min(...completed.map((result) => result.totalMs ?? Infinity)) : null;

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6">
            <h1 className="text-2xl md:text-3xl font-bold">
              Model <span className="gradient-text">Benchmark</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Run one document and one claim set through several models. Every model uses the identical
              production pipeline — same retrieval, same scoring, same guardrails. Only the model changes.
            </p>
          </div>

          {/* Setup */}
          {!run && (
            <GlassCard hover={false} className="p-6 mb-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h2 className="text-sm font-semibold mb-3">1 · Document</h2>
                  <div
                    onClick={() => {
                      const input = document.createElement("input");
                      input.type = "file";
                      input.accept = ".pdf,.png,.jpg,.jpeg,.webp";
                      input.onchange = (event) => {
                        const f = (event.target as HTMLInputElement).files?.[0];
                        if (f) handleFile(f);
                      };
                      input.click();
                    }}
                    className="glass-light rounded-xl border-2 border-dashed border-border/80 hover:border-primary/50 p-6 text-center cursor-pointer transition-colors"
                  >
                    {file ? (
                      <div className="flex items-center gap-3 text-left">
                        <FileText className="w-8 h-8 text-primary shrink-0" />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-semibold truncate">{file.name}</p>
                          <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</p>
                        </div>
                        <button
                          onClick={(event) => {
                            event.stopPropagation();
                            setFile(null);
                            setPreview(null);
                          }}
                          aria-label="Remove selected document"
                          className="p-1.5 rounded-lg hover:bg-surface-light text-muted-foreground"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <Upload className="w-7 h-7 text-primary mx-auto mb-2" />
                        <p className="text-xs font-semibold">Drop or browse a document</p>
                        <p className="text-[10px] text-muted-foreground mt-1">PDF, PNG, JPEG, WebP · up to 4 MB</p>
                      </>
                    )}
                  </div>

                  <h2 className="text-sm font-semibold mt-5 mb-2">2 · Claims to verify</h2>
                  <textarea
                    value={claimsText}
                    onChange={(event) => setClaimsText(event.target.value)}
                    rows={5}
                    placeholder={"Vendor: Microsoft\nInvoice Total: $12,450"}
                    className="w-full glass-light rounded-xl p-3 text-xs border border-border/60 focus:outline-none focus:border-primary bg-transparent resize-y"
                  />
                </div>

                <div>
                  <h2 className="text-sm font-semibold mb-3">3 · Models</h2>
                  <div className="space-y-2">
                    {targets.map((target) => (
                      <label
                        key={target.id}
                        className={`glass-light rounded-lg p-3 flex items-start gap-3 text-xs ${
                          target.available ? "cursor-pointer hover:bg-surface-light/40" : "opacity-55 cursor-not-allowed"
                        }`}
                      >
                        <input
                          type="checkbox"
                          disabled={!target.available}
                          checked={selected.includes(target.id)}
                          onChange={(event) =>
                            setSelected((current) =>
                              event.target.checked ? [...current, target.id] : current.filter((id) => id !== target.id),
                            )
                          }
                          className="mt-0.5 accent-[hsl(var(--primary))]"
                        />
                        <div className="min-w-0">
                          <div className="font-semibold text-foreground font-mono">{target.label}</div>
                          <div className="text-muted-foreground">{target.vendor}</div>
                          {!target.available && <div className="text-warning mt-0.5">{target.reason}</div>}
                        </div>
                      </label>
                    ))}
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-3 leading-relaxed">
                    Models run sequentially — concurrent vision calls hit provider rate limits and would distort the
                    latency figure, which is one of the things being measured. Up to 4 per run.
                  </p>
                </div>
              </div>

              {error && <p className="text-xs text-danger mt-4">{error}</p>}

              <button
                onClick={start}
                disabled={running || !file}
                className="btn-primary w-full py-3 mt-5 text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed relative z-10"
              >
                {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scale className="w-4 h-4" />}
                {running ? "Running benchmark…" : "Run benchmark"}
              </button>
            </GlassCard>
          )}

          {run && (
            <>
              <div className="flex items-center justify-between gap-4 mb-4">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">{run.fileName}</span> · {run.claimCount} claims ·{" "}
                  {(run.totalMs / 1000).toFixed(1)}s total
                </p>
                <button onClick={() => setRun(null)} className="btn-secondary text-xs py-2 px-3.5">
                  New benchmark
                </button>
              </div>

              <div className="glass-light rounded-xl p-3.5 mb-6 border border-accent/30 flex items-start gap-2.5">
                <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                <p className="text-[11px] text-muted-foreground leading-relaxed">{run.disclaimer}</p>
              </div>

              <GlassCard hover={false} className="p-5 mb-6 overflow-x-auto">
                <table className="w-full text-xs min-w-[720px]">
                  <thead className="text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border/50">
                    <tr>
                      <th className="text-left py-2.5">Model</th>
                      <th className="text-right py-2.5">Trust</th>
                      <th className="text-right py-2.5">Verified</th>
                      <th className="text-right py-2.5">Corrections</th>
                      <th className="text-right py-2.5">Unsupported</th>
                      <th className="text-right py-2.5">Needs review</th>
                      <th className="text-right py-2.5">Decisive</th>
                      <th className="text-right py-2.5">Signals</th>
                      <th className="text-right py-2.5">Blocks read</th>
                      <th className="text-right py-2.5">Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {run.results.map((result) => {
                      if (result.error || result.skipped || !result.available) {
                        return (
                          <tr key={result.id}>
                            <td className="py-2.5 font-mono text-muted-foreground">{result.label}</td>
                            <td colSpan={9} className="py-2.5 text-warning flex items-center gap-1.5">
                              <AlertTriangle className="w-3.5 h-3.5 shrink-0" /> {result.error}
                            </td>
                          </tr>
                        );
                      }
                      return (
                        <tr key={result.id} className="hover:bg-surface-light/30">
                          <td className="py-2.5 font-mono text-foreground font-semibold">
                            <span className="flex items-center gap-1.5">
                              {result.trustScore === bestTrust && <Trophy className="w-3.5 h-3.5 text-warning" />}
                              {result.label}
                            </span>
                          </td>
                          <td className="py-2.5 text-right tabular-nums font-bold text-primary">{result.trustScore}%</td>
                          <td className="py-2.5 text-right tabular-nums text-success">{result.verified}</td>
                          <td className="py-2.5 text-right tabular-nums text-warning">{result.corrections}</td>
                          <td className="py-2.5 text-right tabular-nums text-danger">{result.unsupported}</td>
                          <td className="py-2.5 text-right tabular-nums text-accent">{result.needsReview}</td>
                          <td className="py-2.5 text-right tabular-nums">{result.decisionRate}%</td>
                          <td className="py-2.5 text-right tabular-nums">{result.signalsMeasuredAvg}/5</td>
                          <td className="py-2.5 text-right tabular-nums text-muted-foreground">{result.blocksTranscribed}</td>
                          <td className={`py-2.5 text-right tabular-nums ${result.totalMs === fastest ? "text-success font-bold" : "text-muted-foreground"}`}>
                            {((result.totalMs ?? 0) / 1000).toFixed(1)}s
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </GlassCard>

              {/* Where models disagree is the most useful output of a benchmark. */}
              <GlassCard hover={false} className="p-5">
                <h2 className="text-sm font-semibold mb-1">Per-claim agreement</h2>
                <p className="text-[11px] text-muted-foreground mb-4">
                  Claims where models reached different verdicts are the ones worth a human's attention.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs min-w-[560px]">
                    <thead className="text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border/50">
                      <tr>
                        <th className="text-left py-2">Claim</th>
                        {completed.map((result) => (
                          <th key={result.id} className="text-left py-2 font-mono">{result.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40">
                      {(completed[0]?.claims || []).map((claim, index) => {
                        const verdicts = completed.map((result) => result.claims?.[index]);
                        const disagree = new Set(verdicts.map((verdict) => verdict?.status)).size > 1;
                        return (
                          <tr key={claim.field} className={disagree ? "bg-warning/5" : ""}>
                            <td className="py-2.5 font-semibold text-foreground">
                              {claim.field}
                              {disagree && <span className="ml-2 text-[10px] text-warning font-bold uppercase">disagreement</span>}
                            </td>
                            {verdicts.map((verdict, position) => (
                              <td key={position} className="py-2.5">
                                <span
                                  className={
                                    verdict?.status === "verified" ? "text-success"
                                      : verdict?.status === "corrected" ? "text-warning"
                                      : verdict?.status === "unsupported" ? "text-danger" : "text-accent"
                                  }
                                >
                                  {verdict?.status.replace("_", " ") ?? "—"}
                                </span>
                                <span className="text-muted-foreground ml-1.5 tabular-nums">{verdict?.trustScore}%</span>
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
