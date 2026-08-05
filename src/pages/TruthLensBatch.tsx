import { useCallback, useEffect, useRef, useState } from "react";
import { Layers, Upload, X, Loader2, CheckCircle2, AlertTriangle, Download, StopCircle, FileText, Database } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import { invokeAi } from "@/integrations/aiClient";
import { VerificationResult } from "@/types/truthlens";
import { withPreference } from "@/lib/visionProviders";

interface Job {
  id: string;
  label: string;
  status: string;
  totalDocuments: number;
  completedDocuments: number;
  failedDocuments: number;
}

interface JobItem {
  id: string;
  documentId: string | null;
  fileName: string;
  status: string;
  trustScore: number | null;
  totalClaims: number;
  needsReviewClaims: number;
  errorDetail: string | null;
  position: number;
}

const MAX_FILES = 1000;
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

/**
 * Batch verification.
 *
 * The browser drives the loop: it creates the job server-side, then submits one document at a
 * time and reports each outcome back. There is no worker or message broker in this deployment, so
 * rather than pretend there is a distributed queue, the job record lives on the server (surviving
 * refreshes and giving a consolidated report) while the sequencing happens here. Moving to a real
 * worker later changes this page and one API file; the job schema does not change.
 */
export default function TruthLensBatch() {
  const [files, setFiles] = useState<File[]>([]);
  const [claimsText, setClaimsText] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [items, setItems] = useState<JobItem[]>([]);
  const [results, setResults] = useState<VerificationResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cancelled = useRef(false);
  /*
   * Batch keeps its job record server-side so a run survives a refresh, which means it genuinely
   * needs persistence. Dashboard and Review already say so on load; this page used to accept the
   * whole setup and only fail on Start, which reads as a broken feature rather than an
   * unconfigured one.
   */
  const [storage, setStorage] = useState<{ available: boolean; reason?: string } | null>(null);

  useEffect(() => {
    void invokeAi<{ available: boolean; reason?: string }>("workspace", { action: "status" }).then(({ data }) =>
      setStorage({ available: Boolean(data?.available), reason: data?.reason }),
    );
  }, []);

  const addFiles = useCallback((incoming: FileList | null) => {
    if (!incoming) return;
    const accepted: File[] = [];
    const rejected: string[] = [];
    for (const file of Array.from(incoming)) {
      if (file.size > MAX_UPLOAD_BYTES) rejected.push(`${file.name} (too large)`);
      else if (!/\.(pdf|png|jpe?g|webp)$/i.test(file.name)) rejected.push(`${file.name} (unsupported type)`);
      else accepted.push(file);
    }
    setFiles((current) => [...current, ...accepted].slice(0, MAX_FILES));
    // Silently dropping files would make the batch look complete when it wasn't.
    setError(rejected.length > 0 ? `${rejected.length} file(s) skipped: ${rejected.slice(0, 3).join(", ")}${rejected.length > 3 ? " …" : ""}` : null);
  }, []);

  const readAsDataUrl = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target?.result as string);
      reader.onerror = () => reject(new Error(`${file.name} could not be read.`));
      reader.readAsDataURL(file);
    });

  const start = async () => {
    const upstreamClaims = claimsText
      .split("\n")
      .map((line) => line.split(/:(.+)/))
      .filter(([field, value]) => field?.trim() && value?.trim())
      .map(([field, value]) => ({ field: field.trim(), value: value.trim() }));

    if (files.length === 0) return setError("Add at least one document.");
    if (upstreamClaims.length === 0) return setError("Paste the claims to verify against every document.");

    setRunning(true);
    setError(null);
    setResults([]);
    cancelled.current = false;

    const created = await invokeAi<{ job: Job; items: JobItem[] }>("batch-job", {
      action: "create",
      label: `${files.length} document batch`,
      files: files.map((file) => file.name),
    });
    if (created.error || !created.data) {
      setError(created.error?.message ?? "The batch job could not be created.");
      setRunning(false);
      return;
    }

    let currentJob = created.data.job;
    setJob(currentJob);
    setItems(created.data.items);

    for (const item of created.data.items) {
      if (cancelled.current) break;
      const file = files[item.position];
      if (!file) continue;

      setItems((current) => current.map((row) => (row.id === item.id ? { ...row, status: "processing" } : row)));

      try {
        const image = await readAsDataUrl(file);
        const { data, error } = await invokeAi<VerificationResult>(
          "verify-document",
          withPreference({ image, fileName: file.name, upstreamClaims, jobId: currentJob.id }),
        );
        if (error || !data) throw new Error(error?.message ?? "Verification returned no result.");

        setResults((current) => [...current, data]);
        const progress = await invokeAi<{ job: Job; items: JobItem[] }>("batch-job", {
          action: "item",
          jobId: currentJob.id,
          itemId: item.id,
          status: "completed",
          documentId: data.documentId,
          trustScore: data.summary.trustScore,
          totalClaims: data.summary.totalClaims,
          needsReviewClaims: data.summary.needsReviewCount,
        });
        if (progress.data) {
          currentJob = progress.data.job;
          setJob(progress.data.job);
          setItems(progress.data.items);
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Unknown error";
        const progress = await invokeAi<{ job: Job; items: JobItem[] }>("batch-job", {
          action: "item",
          jobId: currentJob.id,
          itemId: item.id,
          status: "failed",
          errorDetail: detail,
        });
        if (progress.data) {
          currentJob = progress.data.job;
          setJob(progress.data.job);
          setItems(progress.data.items);
        }
      }
    }

    if (cancelled.current) {
      const cancelledJob = await invokeAi<{ job: Job; items: JobItem[] }>("batch-job", { action: "cancel", jobId: currentJob.id });
      if (cancelledJob.data) {
        setJob(cancelledJob.data.job);
        setItems(cancelledJob.data.items);
      }
    }
    setRunning(false);
  };

  /** One consolidated record for the whole batch, not one file per document. */
  const exportConsolidated = () => {
    const payload = {
      job,
      generatedAt: new Date().toISOString(),
      documents: results.map((result) => ({
        fileName: result.fileName,
        documentId: result.documentId,
        documentType: result.documentType,
        summary: result.summary,
        verificationTimeMs: result.verificationTimeMs,
        claims: result.claims.map((claim) => ({
          field: claim.field,
          originalValue: claim.originalValue,
          verifiedValue: claim.verifiedValue ?? null,
          status: claim.status,
          trustScore: claim.trustScore,
          signalsMeasured: claim.confidenceBreakdown.measuredCount,
          preVerificationRisk: claim.hallucinationRisk?.level ?? null,
          evidenceCited: claim.retrieval?.citedCount ?? 0,
          reason: claim.reason,
        })),
      })),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `truthlens_batch_${job?.id ?? "report"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const progressPct = job && job.totalDocuments > 0 ? Math.round(((job.completedDocuments + job.failedDocuments) / job.totalDocuments) * 100) : 0;
  const aggregateTrust =
    results.length > 0 ? Math.round(results.reduce((sum, result) => sum + result.summary.trustScore, 0) / results.length) : null;

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6">
            <h1 className="text-2xl md:text-3xl font-bold">
              Batch <span className="gradient-text">Verification</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Verify the same claim set across many documents. Progress is tracked server-side, so the job survives a
              refresh and produces one consolidated report.
            </p>
          </div>

          {storage && !storage.available && (
            <GlassCard hover={false} className="p-10 text-center mb-6">
              <Database className="w-10 h-10 text-primary mx-auto mb-4" />
              <h2 className="text-lg font-bold">Batch verification needs a stored workspace</h2>
              <p className="text-sm text-muted-foreground mt-2 leading-relaxed max-w-xl mx-auto">
                {storage.reason} A batch tracks its progress server-side so the job survives a refresh and produces one
                consolidated report — that requires persistence. Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY,
                then apply the migrations in supabase/migrations.
              </p>
              <p className="text-xs text-muted-foreground mt-4">
                Single-document verification works without any of this — use the Verify page.
              </p>
            </GlassCard>
          )}

          {!job && storage?.available && (
            <GlassCard hover={false} className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h2 className="text-sm font-semibold mb-3">Documents ({files.length})</h2>
                  <div
                    onClick={() => {
                      const input = document.createElement("input");
                      input.type = "file";
                      input.multiple = true;
                      input.accept = ".pdf,.png,.jpg,.jpeg,.webp";
                      input.onchange = (event) => addFiles((event.target as HTMLInputElement).files);
                      input.click();
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      addFiles(event.dataTransfer.files);
                    }}
                    className="glass-light rounded-xl border-2 border-dashed border-border/80 hover:border-primary/50 p-6 text-center cursor-pointer transition-colors"
                  >
                    <Upload className="w-7 h-7 text-primary mx-auto mb-2" />
                    <p className="text-xs font-semibold">Drop or browse documents</p>
                    <p className="text-[10px] text-muted-foreground mt-1">PDF, PNG, JPEG, WebP · 4 MB each · up to {MAX_FILES}</p>
                  </div>

                  {files.length > 0 && (
                    <div className="mt-3 max-h-52 overflow-y-auto space-y-1.5">
                      {files.map((file, index) => (
                        <div key={`${file.name}-${index}`} className="glass-light rounded-lg px-3 py-2 flex items-center gap-2 text-xs">
                          <FileText className="w-3.5 h-3.5 text-primary shrink-0" />
                          <span className="truncate flex-1">{file.name}</span>
                          <span className="text-muted-foreground shrink-0">{(file.size / 1024).toFixed(0)} KB</span>
                          <button
                            onClick={() => setFiles((current) => current.filter((_, position) => position !== index))}
                            aria-label={`Remove ${file.name}`}
                            className="p-1 rounded hover:bg-surface-light text-muted-foreground shrink-0"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div>
                  <h2 className="text-sm font-semibold mb-3">Claims to verify in every document</h2>
                  <textarea
                    value={claimsText}
                    onChange={(event) => setClaimsText(event.target.value)}
                    rows={10}
                    placeholder={"Vendor: Microsoft\nInvoice Total: $12,450\nPayment Terms: Net 30"}
                    className="w-full glass-light rounded-xl p-3 text-xs border border-border/60 focus:outline-none focus:border-primary bg-transparent resize-y"
                  />
                  <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
                    The same claims are checked against each document independently. Documents are processed one at a
                    time to stay inside provider rate limits.
                  </p>
                </div>
              </div>

              {error && <p className="text-xs text-danger mt-4">{error}</p>}

              <button
                onClick={start}
                disabled={running || files.length === 0}
                className="btn-primary w-full py-3 mt-5 text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed relative z-10"
              >
                <Layers className="w-4 h-4" /> Start batch ({files.length} document{files.length === 1 ? "" : "s"})
              </button>
            </GlassCard>
          )}

          {job && (
            <>
              <GlassCard hover={false} className="p-5 mb-6">
                <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
                  <div>
                    <span className="text-[10px] uppercase font-bold tracking-widest text-primary">Job {job.id.slice(0, 8)}</span>
                    <h2 className="text-lg font-bold">{job.label}</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    {running && (
                      <button onClick={() => (cancelled.current = true)} className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5">
                        <StopCircle className="w-3.5 h-3.5 text-danger" /> Stop after current
                      </button>
                    )}
                    {!running && results.length > 0 && (
                      <button onClick={exportConsolidated} className="btn-primary text-xs py-2 px-3.5 flex items-center gap-1.5">
                        <Download className="w-3.5 h-3.5" /> Consolidated report
                      </button>
                    )}
                    {!running && (
                      <button
                        onClick={() => {
                          setJob(null);
                          setItems([]);
                          setFiles([]);
                          setResults([]);
                        }}
                        className="btn-secondary text-xs py-2 px-3.5"
                      >
                        New batch
                      </button>
                    )}
                  </div>
                </div>

                <div className="h-2 rounded-full bg-surface-dark overflow-hidden mb-2">
                  <div className="h-full bg-primary transition-all duration-500" style={{ width: `${progressPct}%` }} />
                </div>
                <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                  <span className="font-bold text-foreground tabular-nums">{progressPct}%</span>
                  <span>{job.completedDocuments} completed</span>
                  {job.failedDocuments > 0 && <span className="text-danger">{job.failedDocuments} failed</span>}
                  <span>of {job.totalDocuments}</span>
                  {aggregateTrust !== null && <span className="text-primary font-semibold">Batch average trust {aggregateTrust}%</span>}
                  <span className="capitalize">· {job.status}</span>
                </div>
              </GlassCard>

              <GlassCard hover={false} className="p-5 overflow-x-auto">
                <table className="w-full text-xs min-w-[560px]">
                  <thead className="text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border/50">
                    <tr>
                      <th className="text-left py-2.5">Document</th>
                      <th className="text-left py-2.5">Status</th>
                      <th className="text-right py-2.5">Trust</th>
                      <th className="text-right py-2.5">Claims</th>
                      <th className="text-right py-2.5">Needs review</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {items.map((item) => (
                      <tr key={item.id} className="hover:bg-surface-light/30">
                        <td className="py-2.5 text-foreground truncate max-w-xs">{item.fileName}</td>
                        <td className="py-2.5">
                          <span className="flex items-center gap-1.5">
                            {item.status === "completed" && <CheckCircle2 className="w-3.5 h-3.5 text-success" />}
                            {item.status === "failed" && <AlertTriangle className="w-3.5 h-3.5 text-danger" />}
                            {item.status === "processing" && <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />}
                            <span
                              className={
                                item.status === "completed" ? "text-success"
                                  : item.status === "failed" ? "text-danger"
                                  : item.status === "processing" ? "text-primary" : "text-muted-foreground"
                              }
                            >
                              {item.status}
                            </span>
                          </span>
                          {item.errorDetail && <p className="text-[10px] text-danger mt-0.5">{item.errorDetail}</p>}
                        </td>
                        <td className="py-2.5 text-right tabular-nums font-bold text-primary">{item.trustScore !== null ? `${item.trustScore}%` : "—"}</td>
                        <td className="py-2.5 text-right tabular-nums">{item.totalClaims || "—"}</td>
                        <td className="py-2.5 text-right tabular-nums text-accent">{item.needsReviewClaims || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </GlassCard>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
