import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  X,
  Loader2,
  Shield,
  Search,
  ArrowRight,
  History,
  Sparkles,
  ShieldCheck,
  RefreshCw,
  Lock,
  Cpu,
  Wand2,
  Info,
} from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import VerificationCard from "@/components/truthlens/VerificationCard";
import VerificationSidePanel from "@/components/truthlens/VerificationSidePanel";
import AuditTrailDrawer from "@/components/truthlens/AuditTrailDrawer";
import VisionProviderSelector from "@/components/truthlens/VisionProviderSelector";
import VerificationSummaryCards from "@/components/truthlens/VerificationSummaryCards";
import PipelineTimeline from "@/components/truthlens/PipelineTimeline";
import ClaimRelationGraph from "@/components/truthlens/ClaimRelationGraph";
import { invokeAi } from "@/integrations/aiClient";
import { Claim, ClaimStatus, HumanFeedback, VerificationResult } from "@/types/truthlens";
import { withPreference } from "@/lib/visionProviders";
import { prepareDocument, ACCEPTED_EXTENSIONS, ACCEPTED_PATTERN, type PreparedDocument } from "@/lib/documentInput";

type Phase = "upload" | "processing" | "results";

const MAX_UPLOAD_BYTES = 8 * 1024 * 1024; // PDFs are rasterised client-side, so the source may be larger

export default function TruthLensVerify() {
  const [phase, setPhase] = useState<Phase>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [selectedClaim, setSelectedClaim] = useState<Claim | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Filters & Search
  const [statusFilter, setStatusFilter] = useState<"all" | ClaimStatus>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [isAuditTrailOpen, setIsAuditTrailOpen] = useState(false);
  const [upstreamClaimsText, setUpstreamClaimsText] = useState("");
  const [verificationError, setVerificationError] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  /** Kept separate from verificationError so it survives into the results view. */
  const [claimParseWarning, setClaimParseWarning] = useState<string | null>(null);
  const [extracting, setExtracting] = useState(false);
  /** Set when the claims came from the document itself rather than another AI system. */
  const [selfExtracted, setSelfExtracted] = useState(false);
  /** Pages rendered from the upload. A PDF becomes images here so the pipeline sees only images. */
  const [prepared, setPrepared] = useState<PreparedDocument | null>(null);
  const [preparing, setPreparing] = useState(false);

  // Everything downstream is an image: page 1 of the prepared document drives both the thumbnail
  // and the evidence viewer, so there is no PDF branch anywhere past intake.
  const imagePreview = prepared?.pages[0]?.dataUrl ?? null;

  const handleFile = useCallback(async (f: File) => {
    setVerificationError(null);
    if (!ACCEPTED_PATTERN.test(f.name)) {
      setVerificationError("Unsupported file type. Upload a PDF, PNG, JPG, JPEG, WebP or TIFF document.");
      return;
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      setVerificationError(`${(f.size / 1024 / 1024).toFixed(1)} MB exceeds the 8 MB limit. Reduce the resolution or split the document.`);
      return;
    }
    setFile(f);
    setPreparing(true);
    try {
      const document = await prepareDocument(f);
      setPrepared(document);
      setPreview(document.pages[0]?.dataUrl ?? null);
    } catch (error) {
      setVerificationError(error instanceof Error ? error.message : "This document could not be prepared.");
      setFile(null);
      setPrepared(null);
    } finally {
      setPreparing(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) void handleFile(f);
    },
    [handleFile]
  );

  /**
   * Self-check mode: let TruthLens propose the document's business facts, then verify them
   * through the ordinary pipeline. The proposals land in the same editable textarea as pasted
   * claims — they are candidates for the user to correct, never an answer.
   */
  const extractClaims = useCallback(async () => {
    if (!file || !prepared) return;
    setExtracting(true);
    setVerificationError(null);

    const page = prepared?.pages[0];
    if (!page) return;
    const { data, error } = await invokeAi<{ claims: Array<{ field: string; value: string }>; caveat: string }>(
      "extract-claims",
      withPreference({ image: page.dataUrl, fileName: prepared.fileName }),
    );

    if (error || !data?.claims?.length) {
      setVerificationError(error?.message || "No business facts could be extracted. Paste the claims manually instead.");
    } else {
      setUpstreamClaimsText(data.claims.map((claim) => `${claim.field}: ${claim.value}`).join("\n"));
      setSelfExtracted(true);
    }
    setExtracting(false);
  }, [file, prepared]);

  const startVerification = useCallback(async () => {
    if (!file) return;

    const upstreamClaims: { field: string; value: string }[] = [];
    const malformed: string[] = [];
    for (const line of upstreamClaimsText.split("\n")) {
      if (!line.trim()) continue;
      const [field, value] = line.split(/:(.+)/);
      if (field?.trim() && value?.trim()) upstreamClaims.push({ field: field.trim(), value: value.trim() });
      else malformed.push(line.trim());
    }
    if (upstreamClaims.length === 0) {
      setVerificationError("Paste the upstream AI claims first, one per line in the format Field: Value.");
      return;
    }
    // Previously these lines were dropped in silence, so a user could believe a claim was
    // verified when it was never sent.
    setClaimParseWarning(
      malformed.length > 0
        ? `${malformed.length} line(s) were not verified — no "Field: Value" separator: ${malformed.slice(0, 3).join(" · ")}${malformed.length > 3 ? " …" : ""}`
        : null,
    );

    setPhase("processing");
    setReviewError(null);

    const page = prepared?.pages[0];
    if (!page) {
      setVerificationError("The document is still being prepared. Try again in a moment.");
      setPhase("upload");
      return;
    }

    const { data, error } = await invokeAi<VerificationResult>(
      "verify-document",
      withPreference({
        image: page.dataUrl,
        fileName: prepared.fileName,
        upstreamClaims,
        // Tells the server which verification path this is, so the result records it honestly.
        selfExtracted,
      }),
    );

    if (error || !data?.claims?.length) {
      setVerificationError(error?.message || "The verification service did not return an evidence-backed result. No fallback result was created.");
      setPhase("upload");
      return;
    }

    setResult(data);
    setSelectedClaim(null);
    setVerificationError(null);
    setPhase("results");
  }, [file, prepared, upstreamClaimsText, selfExtracted]);

  const reset = () => {
    setPhase("upload");
    setFile(null);
    setPreview(null);
    setPrepared(null);
    setResult(null);
    setSelectedClaim(null);
    setStatusFilter("all");
    setSearchQuery("");
    setUpstreamClaimsText("");
    setVerificationError(null);
    setReviewError(null);
    setClaimParseWarning(null);
    setSelfExtracted(false);
  };

  /**
   * Human decisions are only meaningful against persisted claims — an unpersisted run has no
   * durable claim id to attach the decision to. The previous version fired the write blind and
   * showed success regardless, while every request failed server-side.
   */
  const handleFeedbackUpdate = async (claimId: string, feedback: HumanFeedback) => {
    if (!result) return;
    if (!result.persistence.persisted || !result.documentId) {
      setReviewError(result.persistence.reason || "This verification was not stored, so review decisions cannot be recorded.");
      return;
    }

    setReviewError(null);
    const { error } = await invokeAi("review-claim", { documentId: result.documentId, claimId, feedback });
    if (error) {
      setReviewError(`Review decision was not recorded: ${error.message}`);
      return;
    }

    const applied: HumanFeedback = { ...feedback, timestamp: new Date().toISOString() };
    const updatedClaims = result.claims.map((c) =>
      c.id === claimId
        ? {
            ...c,
            feedback: applied,
            ...(applied.status === "overridden" && applied.overrideValue
              ? { verifiedValue: applied.overrideValue, status: "corrected" as ClaimStatus }
              : {}),
          }
        : c,
    );
    setResult({ ...result, claims: updatedClaims });
    setSelectedClaim((current) => (current?.id === claimId ? updatedClaims.find((c) => c.id === claimId) || null : current));
  };

  const filteredClaims = (result?.claims || []).filter((c) => {
    if (statusFilter !== "all" && c.status !== statusFilter) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        c.field.toLowerCase().includes(q) ||
        c.originalValue.toLowerCase().includes(q) ||
        Boolean(c.verifiedValue?.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const reviewEnabled = Boolean(result?.persistence.persisted);

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />

      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-16 px-4 md:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Header Bar with Vision Provider Selector */}
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4 pb-6 border-b border-border/40">
            <div>
              <div className="inline-flex items-center gap-2 glass rounded-full px-3 py-1 text-xs text-primary mb-2">
                <Sparkles className="w-3.5 h-3.5 text-accent" /> Enterprise AI Hallucination Verification Engine
              </div>
              <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
                Universal <span className="gradient-text">Verification Studio</span>
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Upload ANY document type — Resumes, Contracts, Medical Reports, Diagrams, Financials, POs.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <VisionProviderSelector />
              {phase === "results" && result && (
                <button
                  onClick={() => setIsAuditTrailOpen(true)}
                  className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-2"
                >
                  <History className="w-4 h-4 text-primary" /> Compliance Audit Trail
                </button>
              )}
            </div>
          </div>

          <AnimatePresence mode="wait">
            {/* PHASE 1: UPLOAD */}
            {phase === "upload" && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-2xl mx-auto"
              >
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  className={`glass rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 min-h-[320px] flex flex-col items-center justify-center border-2 border-dashed ${
                    dragOver
                      ? "border-primary bg-primary/10 scale-[1.01]"
                      : file
                      ? "border-success/40 bg-success/5"
                      : "border-border/80 hover:border-primary/50 hover:bg-surface-light/30"
                  }`}
                  onClick={() => {
                    if (!file) {
                      const input = document.createElement("input");
                      input.type = "file";
                      input.accept = ACCEPTED_EXTENSIONS;
                      input.onchange = (e) => {
                        const f = (e.target as HTMLInputElement).files?.[0];
                        if (f) void handleFile(f);
                      };
                      input.click();
                    }
                  }}
                >
                  {!file ? (
                    <>
                      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 text-primary animate-pulse-glow">
                        <Upload className="w-8 h-8" />
                      </div>
                      <p className="text-xl font-bold mb-2">Drop document to verify claims</p>
                      <p className="text-xs text-muted-foreground mb-6 max-w-sm">
                        PDF, PNG, JPG, WebP or TIFF · up to 8 MB · any document type
                      </p>
                      <span className="btn-secondary py-2 px-5 text-xs font-semibold">Browse File</span>
                    </>
                  ) : (
                    <div className="w-full">
                      <div className="flex items-center gap-4 mb-6 glass-light p-4 rounded-xl text-left border border-border">
                        {imagePreview ? (
                          <img src={imagePreview} alt="Preview" className="w-20 h-20 object-cover rounded-xl border border-border" />
                        ) : (
                          <div className="w-20 h-20 rounded-xl bg-surface-light flex flex-col items-center justify-center shrink-0 gap-1">
                            <FileText className="w-7 h-7 text-primary" />
                            
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="font-bold truncate text-base">{file.name}</p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            {(file.size / 1024).toFixed(1)} KB
                            {prepared?.converted && ` · ${prepared.pages.length} page(s) rendered to images`}
                          </p>
                          {prepared?.truncatedFrom && (
                            <p className="text-[10px] text-warning mt-1">
                              Only the first {prepared.pages.length} of {prepared.truncatedFrom} pages will be verified.
                            </p>
                          )}
                          <div className="inline-flex items-center gap-1 text-[10px] uppercase font-bold text-success bg-success/10 px-2 py-0.5 rounded mt-2">
                            {preparing ? (
                              <>Preparing document…</>
                            ) : (
                              <>
                                <ShieldCheck className="w-3 h-3" /> Ready to verify
                              </>
                            )}
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setFile(null);
                            setPreview(null);
                          }}
                          aria-label="Remove selected document"
                          className="p-2 rounded-lg hover:bg-surface-light text-muted-foreground"
                        >
                          <X className="w-5 h-5" />
                        </button>
                      </div>

                      {/* The mode is stated before the user types, not discovered afterwards. */}
                      <div
                        className={`text-left mb-3 rounded-xl p-3 border ${
                          selfExtracted ? "border-warning/40 bg-warning/5" : "border-primary/30 bg-primary/5"
                        }`}
                      >
                        <p className="text-[11px] font-bold uppercase tracking-wider flex items-center gap-1.5">
                          {selfExtracted ? (
                            <>
                              <Wand2 className="w-3.5 h-3.5 text-warning" />
                              <span className="text-warning">Self-check mode</span>
                            </>
                          ) : (
                            <>
                              <ShieldCheck className="w-3.5 h-3.5 text-primary" />
                              <span className="text-primary">Cross-check mode</span>
                            </>
                          )}
                        </p>
                        <p className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
                          {selfExtracted
                            ? "TruthLens proposed these claims itself. Both passes share a failure mode, so this is weaker evidence — edit anything that looks wrong, or paste an external AI's output instead."
                            : "Paste what ChatGPT, Claude, Gemini or any vision model said about this document. TruthLens independently retrieves evidence and never trusts those claims."}
                        </p>
                      </div>

                      <div className="text-left mb-4">
                        <div className="flex items-center justify-between gap-3 mb-1.5">
                          <label className="text-xs font-semibold text-foreground">Claims to verify</label>
                          <button
                            onClick={(event) => {
                              event.stopPropagation();
                              void extractClaims();
                            }}
                            disabled={extracting}
                            className="btn-secondary text-[11px] py-1.5 px-2.5 flex items-center gap-1.5 disabled:opacity-50"
                          >
                            {extracting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3 h-3 text-accent" />}
                            {extracting ? "Extracting…" : "Extract from document"}
                          </button>
                        </div>
                        <textarea
                          value={upstreamClaimsText}
                          onChange={(event) => {
                            setUpstreamClaimsText(event.target.value);
                            setSelfExtracted(false);
                          }}
                          onClick={(event) => event.stopPropagation()}
                          rows={6}
                          placeholder={"Vendor: Microsoft\nInvoice Total: $12,450\nPayment Terms: Net 30"}
                          className="w-full glass-light rounded-xl p-3 text-xs border border-border/60 focus:outline-none focus:border-primary bg-transparent resize-y font-mono"
                        />
                        <p className="text-[10px] text-muted-foreground mt-1.5">
                          One claim per line as <span className="font-mono">Field: Value</span>. Paste what your AI system
                          said about this document — TruthLens checks those statements against evidence and will not invent claims.
                        </p>

                      </div>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startVerification();
                        }}
                        disabled={preparing || !prepared}
                        className="btn-primary w-full py-4 text-base flex items-center justify-center gap-2 relative z-10 font-bold disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <Shield className="w-5 h-5" /> Execute Enterprise Verification <ArrowRight className="w-5 h-5" />
                      </button>
                    </div>
                  )}
                </div>
                {verificationError && <p className="mt-4 text-xs text-danger text-center">{verificationError}</p>}
              </motion.div>
            )}

            {/* PHASE 2: PROCESSING — a single honest indeterminate state. Stage timings are
                measured server-side and shown with the result; nothing is simulated here. */}
            {phase === "processing" && (
              <motion.div
                key="processing"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="max-w-lg mx-auto text-center"
              >
                <div className="w-20 h-20 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center mx-auto mb-6">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                </div>
                <h2 className="text-2xl font-bold mb-2">Verifying against document evidence</h2>
                <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto leading-relaxed">
                  {selfExtracted
                    ? "Self-check: the claims TruthLens proposed are being checked against independently retrieved evidence."
                    : "Cross-check: your AI's claims are being checked against evidence retrieved independently from the document."}
                </p>

                {/* Indeterminate, but shaped like the real pipeline so the wait reads as progress. */}
                <div className="h-1 rounded-full bg-surface-dark overflow-hidden progress-sweep mb-6" role="progressbar" aria-label="Verification in progress" />

                <GlassCard hover={false} className="p-5 text-left space-y-2.5">
                  {[
                    "Reading the page — text, coordinates and confidence",
                    "Searching the document for evidence, without a model",
                    "Predicting hallucination risk before any verdict",
                    "Checking each claim against retrieved evidence only",
                  ].map((step) => (
                    <div key={step} className="flex items-start gap-2.5 text-[11px] text-muted-foreground">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary/60 mt-1.5 shrink-0" />
                      <span className="leading-relaxed">{step}</span>
                    </div>
                  ))}
                </GlassCard>
                <p className="text-[11px] text-muted-foreground mt-4 flex items-center justify-center gap-2">
                  <Cpu className="w-3.5 h-3.5 text-primary shrink-0" />
                  Any claim without retrievable evidence is returned as
                  <span className="text-accent font-semibold">needs review</span>, never a guess.
                </p>
              </motion.div>
            )}

            {/* PHASE 3: RESULTS */}
            {phase === "results" && result && (
              <motion.div key="results" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                {/* Results Action Bar */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4 glass-light p-4 rounded-2xl border border-border/50">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-primary tracking-widest block">Verified Document Context</span>
                    <div className="flex items-center gap-2 flex-wrap">
                      <h2 className="text-lg font-bold text-foreground break-anywhere">{result.fileName}</h2>
                      <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-primary/20 text-primary border border-primary/30">
                        {result.documentType}
                      </span>
                      <span
                        className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-surface-light text-muted-foreground border border-border"
                        title={result.providerLabel}
                      >
                        {result.modelUsed}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border ${
                          result.verificationMode === "self-check"
                            ? "bg-warning/15 text-warning border-warning/30"
                            : "bg-primary/15 text-primary border-primary/30"
                        }`}
                        title={
                          result.verificationMode === "self-check"
                            ? "Claims were proposed by TruthLens itself — weaker evidence than checking another system."
                            : "Claims came from an external AI system and were independently verified."
                        }
                      >
                        {result.verificationMode === "self-check" ? "Self-check" : "Cross-check"}
                      </span>
                      {result.ocr && (
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                            result.ocr.engine === "paddleocr"
                              ? "bg-success/10 text-success border-success/30"
                              : "bg-surface-light text-muted-foreground border-border"
                          }`}
                          title={result.ocr.degradedReason ?? "Deterministic OCR: the same page always yields the same text and coordinates."}
                        >
                          {result.ocr.engine === "paddleocr" ? "PaddleOCR" : "model OCR"}
                        </span>
                      )}
                      {result.failover && result.failover.length > 0 && (
                        <span
                          className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-warning/15 text-warning border border-warning/30"
                          title={`Preferred provider failed: ${result.failover.join(" | ")}`}
                        >
                          failover
                        </span>
                      )}
                    </div>
                  </div>
                  <button onClick={reset} className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5">
                    <RefreshCw className="w-3.5 h-3.5" /> Verify Another Document
                  </button>
                </div>

                {/* Persistence state — determines whether human review is possible at all */}
                {!reviewEnabled && result.persistence.reason && (
                  <div className="mb-6 glass-light rounded-xl p-3.5 border border-warning/40 flex items-start gap-2.5">
                    <Lock className="w-4 h-4 text-warning shrink-0 mt-0.5" />
                    <div className="text-xs">
                      <span className="font-semibold text-warning">Demo mode · not stored</span>
                      <p className="text-muted-foreground mt-0.5 leading-relaxed">{result.persistence.reason}</p>
                    </div>
                  </div>
                )}
                {claimParseWarning && (
                  <div className="mb-6 glass-light rounded-xl p-3.5 border border-warning/40 text-xs text-warning">{claimParseWarning}</div>
                )}
                {reviewError && (
                  <div className="mb-6 glass-light rounded-xl p-3.5 border border-danger/40 text-xs text-danger">{reviewError}</div>
                )}

                <VerificationSummaryCards summary={result.summary} />

                <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6 items-start">
                  <div>
                    {/* Filters & Search Toolbar */}
                    <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between mb-6 gap-3">
                      <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                        {(["all", "verified", "corrected", "unsupported", "needs_review"] as const).map((st) => (
                          <button
                            key={st}
                            onClick={() => setStatusFilter(st)}
                            className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all whitespace-nowrap capitalize ${
                              statusFilter === st
                                ? "bg-primary text-primary-foreground border-primary shadow-sm"
                                : "glass-light border-border text-muted-foreground hover:text-foreground"
                            }`}
                          >
                            {st.replace("_", " ")}
                          </button>
                        ))}
                      </div>

                      <div className="relative">
                        <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                        <input
                          type="text"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          placeholder="Search claims by field or value..."
                          className="glass-light pl-9 pr-4 py-1.5 rounded-xl text-xs border border-border/60 focus:outline-none focus:border-primary w-full sm:w-64"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                      {filteredClaims.map((claim) => (
                        <VerificationCard
                          key={claim.id}
                          claim={claim}
                          isSelected={selectedClaim?.id === claim.id}
                          onSelect={() => setSelectedClaim(claim)}
                          onFeedback={handleFeedbackUpdate}
                          reviewEnabled={reviewEnabled}
                        />
                      ))}
                    </div>

                    {filteredClaims.length === 0 && (
                      <div className="glass rounded-2xl p-12 text-center text-muted-foreground">
                        <Search className="w-10 h-10 mx-auto mb-3 text-muted-foreground/50" />
                        <p className="text-sm font-semibold">No claims match the active filter</p>
                      </div>
                    )}
                  </div>

                  <div className="space-y-6">
                    <PipelineTimeline events={result.timeline} totalMs={result.verificationTimeMs} />
                    <ClaimRelationGraph
                      claims={result.claims}
                      relations={result.relations || []}
                      selectedClaimId={selectedClaim?.id}
                      onSelectClaim={setSelectedClaim}
                    />
                  </div>
                </div>

                <VerificationSidePanel
                  claim={selectedClaim}
                  documentSource={preview}
                  fileName={result.fileName}
                  reviewEnabled={reviewEnabled}
                  onClose={() => setSelectedClaim(null)}
                  onFeedbackUpdate={handleFeedbackUpdate}
                />

                <AuditTrailDrawer isOpen={isAuditTrailOpen} onClose={() => setIsAuditTrailOpen(false)} result={result} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
