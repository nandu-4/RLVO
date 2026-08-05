import { lazy, Suspense, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Shield,
  Eye,
  Clock,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ShieldAlert,
  ArrowRight,
  Layers,
  Sparkles,
  Search,
  ThumbsUp,
  ThumbsDown,
  Edit3,
  GitCompare,
  Brain,
  Activity,
  Radar,
  Loader2,
} from "lucide-react";
import { Claim, Evidence, HumanFeedback } from "@/types/truthlens";
import GlassCard from "./GlassCard";
import TrustScoreBreakdown from "./TrustScoreBreakdown";

// pdf.js is ~1.3MB. Load the viewer only when the Evidence tab is opened, so it never sits in
// the initial bundle for users who only read the summary.
const DocumentEvidenceViewer = lazy(() => import("./DocumentEvidenceViewer"));

type DrawerTab = "overview" | "evidence" | "reasoning" | "comparison" | "timeline";

interface VerificationSidePanelProps {
  claim: Claim | null;
  documentSource: string | null;
  fileName: string;
  /** False when the run was not persisted — a decision would have nothing durable to attach to. */
  reviewEnabled: boolean;
  onClose: () => void;
  onFeedbackUpdate: (claimId: string, feedback: HumanFeedback) => void | Promise<void>;
}

const TABS: { id: DrawerTab; label: string; icon: React.ReactNode }[] = [
  { id: "overview", label: "Overview", icon: <Shield className="w-3.5 h-3.5" /> },
  { id: "evidence", label: "Evidence", icon: <Eye className="w-3.5 h-3.5" /> },
  { id: "reasoning", label: "Reasoning", icon: <Brain className="w-3.5 h-3.5" /> },
  { id: "comparison", label: "Comparison", icon: <GitCompare className="w-3.5 h-3.5" /> },
  { id: "timeline", label: "Timeline", icon: <Clock className="w-3.5 h-3.5" /> },
];

const REASONING_STEP_ICONS: Record<string, { icon: React.ReactNode; color: string }> = {
  Planning: { icon: <Search className="w-4 h-4" />, color: "text-accent border-accent" },
  "Evidence Search": { icon: <Eye className="w-4 h-4" />, color: "text-primary border-primary" },
  Reflection: { icon: <Brain className="w-4 h-4" />, color: "text-warning border-warning" },
  Decision: { icon: <CheckCircle2 className="w-4 h-4" />, color: "text-success border-success" },
};

const RISK_STYLE = {
  LOW: "border-success/40 bg-success/5 text-success",
  MEDIUM: "border-warning/40 bg-warning/5 text-warning",
  HIGH: "border-danger/40 bg-danger/5 text-danger",
} as const;

export default function VerificationSidePanel({
  claim,
  documentSource,
  fileName,
  reviewEnabled,
  onClose,
  onFeedbackUpdate,
}: VerificationSidePanelProps) {
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [overrideValue, setOverrideValue] = useState("");
  const [reviewerNotes, setReviewerNotes] = useState("");
  const [showOverrideInput, setShowOverrideInput] = useState(false);
  const [activeTab, setActiveTab] = useState<DrawerTab>("overview");

  if (!claim) return null;

  const citedEvidence = claim.evidence.filter((item) => item.cited);
  const activeEv = selectedEvidence || citedEvidence[0] || claim.evidence[0] || null;

  const handleFeedback = (type: "approved" | "rejected" | "overridden") => {
    if (type === "overridden") {
      if (!showOverrideInput) {
        setShowOverrideInput(true);
        return;
      }
      if (!overrideValue.trim()) return;
    }
    void onFeedbackUpdate(claim.id, {
      status: type,
      overrideValue: type === "overridden" ? overrideValue.trim() : undefined,
      reviewerNotes: reviewerNotes.trim() || undefined,
      timestamp: new Date().toISOString(),
    });
    setShowOverrideInput(false);
  };

  const statusColor =
    claim.status === "verified" ? "text-success" :
    claim.status === "corrected" ? "text-warning" :
    claim.status === "unsupported" ? "text-danger" : "text-accent";

  const statusIcon =
    claim.status === "verified" ? <CheckCircle2 className="w-4 h-4" /> :
    claim.status === "corrected" ? <AlertTriangle className="w-4 h-4" /> :
    claim.status === "unsupported" ? <ShieldAlert className="w-4 h-4" /> :
    <HelpCircle className="w-4 h-4" />;

  /* ───────── TAB CONTENT ───────── */

  const renderOverviewTab = () => (
    <div className="space-y-5">
      <TrustScoreBreakdown breakdown={claim.confidenceBreakdown} trustScore={claim.trustScore} />

      {/* Predicted before verification ran — so it is actionable, not a restatement of the score. */}
      {claim.hallucinationRisk && (
        <GlassCard hover={false} className="p-5">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Radar className="w-4 h-4 text-warning" /> Pre-verification hallucination risk
            </h3>
            <span className={`px-2.5 py-1 rounded-full text-[11px] font-black border ${RISK_STYLE[claim.hallucinationRisk.level]}`}>
              {claim.hallucinationRisk.level} · {claim.hallucinationRisk.score}%
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground mb-3 leading-relaxed">
            Computed from document legibility and retrieval strength <em>before</em> the verifier ran — it never sees the outcome.
          </p>
          <ul className="space-y-1.5">
            {claim.hallucinationRisk.reasons.map((reason, index) => (
              <li key={index} className="text-[11px] text-muted-foreground leading-relaxed flex gap-2">
                <span className="text-warning font-bold shrink-0">·</span>
                {reason}
              </li>
            ))}
          </ul>
        </GlassCard>
      )}

      <GlassCard hover={false} className="p-5 space-y-3">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" /> Claim Summary
        </h3>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="glass-light rounded-lg p-3">
            <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Status</span>
            <span className={`font-bold flex items-center gap-1.5 ${statusColor}`}>
              {statusIcon} {claim.status.replace("_", " ").toUpperCase()}
            </span>
          </div>
          <div className="glass-light rounded-lg p-3">
            <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Evidence</span>
            <span className="font-bold text-foreground flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-primary" /> {citedEvidence.length} cited / {claim.evidence.length} retrieved
            </span>
          </div>
          <div className="glass-light rounded-lg p-3 col-span-2">
            <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Decision reason</span>
            <p className="text-muted-foreground leading-relaxed">{claim.reason}</p>
          </div>
        </div>
      </GlassCard>

      {/* Human review */}
      <GlassCard hover={false} className="p-5">
        <h3 className="text-sm font-semibold mb-3">Human-in-the-Loop Decision</h3>
        {!reviewEnabled ? (
          <p className="text-xs text-muted-foreground leading-relaxed">
            Review is unavailable because this verification was not stored. Decisions must attach to a durable claim
            record so they can be audited; sign in to an organization on a configured deployment to enable it.
          </p>
        ) : (
          <>
            {claim.feedback && (
              <div className="glass-light rounded-lg p-3 mb-3 text-xs">
                <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Recorded decision</span>
                <span className="font-bold text-foreground capitalize">{claim.feedback.status}</span>
                {claim.feedback.overrideValue && <span className="text-success font-mono ml-2">→ {claim.feedback.overrideValue}</span>}
                {claim.feedback.reviewerNotes && <p className="text-muted-foreground mt-1 leading-relaxed">{claim.feedback.reviewerNotes}</p>}
              </div>
            )}

            <label className="block text-[10px] uppercase font-bold tracking-wider text-muted-foreground mb-1.5">Reviewer comments</label>
            <textarea
              value={reviewerNotes}
              onChange={(e) => setReviewerNotes(e.target.value)}
              rows={2}
              placeholder="Why are you approving, rejecting, or overriding this claim?"
              className="w-full glass-light px-3 py-2 rounded-lg text-xs border border-border focus:outline-none focus:border-primary bg-transparent resize-y mb-3"
            />

            <div className="flex items-center gap-2 mb-3">
              <button onClick={() => handleFeedback("approved")} className="btn-secondary py-2 px-3 text-xs flex items-center gap-1.5">
                <ThumbsUp className="w-3.5 h-3.5 text-success" /> Approve
              </button>
              <button onClick={() => handleFeedback("rejected")} className="btn-secondary py-2 px-3 text-xs flex items-center gap-1.5">
                <ThumbsDown className="w-3.5 h-3.5 text-danger" /> Reject
              </button>
              <button onClick={() => setShowOverrideInput(!showOverrideInput)} className="btn-primary py-2 px-3 text-xs flex items-center gap-1.5">
                <Edit3 className="w-3.5 h-3.5" /> Override
              </button>
            </div>

            {showOverrideInput && (
              <div className="space-y-2 pt-2 border-t border-border/50">
                <input
                  type="text"
                  value={overrideValue}
                  onChange={(e) => setOverrideValue(e.target.value)}
                  placeholder="Enter the corrected value..."
                  className="w-full glass-light px-3 py-2 rounded-lg text-xs border border-border focus:outline-none focus:border-primary"
                />
                <button
                  onClick={() => handleFeedback("overridden")}
                  disabled={!overrideValue.trim()}
                  className="btn-primary py-1.5 px-4 text-xs w-full disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Save Override
                </button>
              </div>
            )}
          </>
        )}
      </GlassCard>
    </div>
  );

  const renderEvidenceTab = () => (
    <div className="space-y-5">
      {/* What the retrieval engine searched — the audit trail of the search itself. */}
      {claim.retrieval && (
        <GlassCard hover={false} className="p-5">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <Search className="w-4 h-4 text-primary" /> Evidence Retrieval Engine
          </h3>
          <div className="grid grid-cols-2 gap-3 text-xs mb-3">
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Surfaces searched</span>
              <span className="text-foreground">{claim.retrieval.searched.join(" · ") || "Transcribed text"}</span>
            </div>
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Strategies that hit</span>
              <span className="text-foreground">{claim.retrieval.strategies.join(" · ") || "None matched"}</span>
            </div>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            Retrieval ran before the verifier saw this claim, and the verifier could only cite what it returned —
            {" "}{claim.retrieval.citedCount} of {claim.retrieval.candidateCount} candidate(s).
          </p>
        </GlassCard>
      )}

      <GlassCard hover={false} className="p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-accent" />
            <h3 className="text-sm font-semibold">Interactive Document Evidence Viewer</h3>
          </div>
          {activeEv && (
            <span className="text-xs text-muted-foreground">Page {activeEv.pageNumber} · {activeEv.layoutRegion}</span>
          )}
        </div>

        <Suspense
          fallback={
            <div className="w-full h-[30rem] rounded-xl glass-light border border-border/80 flex items-center justify-center">
              <Loader2 className="w-6 h-6 text-primary animate-spin" />
            </div>
          }
        >
          <DocumentEvidenceViewer
            source={documentSource}
            fileName={fileName}
            evidence={claim.evidence}
            activeEvidence={activeEv}
            onSelectEvidence={setSelectedEvidence}
          />
        </Suspense>

        <div className="mt-3 flex items-center gap-2 overflow-x-auto pb-1">
          {claim.evidence.map((ev, i) => (
            <button
              key={ev.id}
              onClick={() => setSelectedEvidence(ev)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all whitespace-nowrap ${
                activeEv?.id === ev.id
                  ? "bg-primary/20 border-primary text-primary"
                  : ev.cited
                  ? "glass-light border-success/40 text-muted-foreground hover:text-foreground"
                  : "glass-light border-border/60 text-muted-foreground/70 hover:text-foreground"
              }`}
            >
              #{i + 1} {ev.source}{!ev.cited && " (uncited)"}
            </button>
          ))}
        </div>
      </GlassCard>

      {activeEv && (
        <GlassCard hover={false} className="p-5">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" /> Evidence Detail
          </h3>
          <div className="space-y-2 text-xs">
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Source</span>
              <span className="text-foreground font-semibold">{activeEv.source}</span>
              <span className={`ml-2 text-[10px] font-bold uppercase ${activeEv.cited ? "text-success" : "text-muted-foreground"}`}>
                {activeEv.cited ? "cited by verifier" : "retrieved, not cited"}
              </span>
            </div>
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Transcribed text</span>
              <span className="text-foreground font-mono leading-relaxed">{activeEv.text}</span>
            </div>
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Found by</span>
              <span className="text-foreground">{activeEv.retrievedBy?.join(" · ") || "—"}</span>
            </div>
            <div className="glass-light rounded-lg p-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">Retrieval score</span>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 rounded-full bg-surface-dark overflow-hidden">
                  <div className="h-full bg-primary" style={{ width: `${activeEv.confidence}%` }} />
                </div>
                <span className="text-foreground font-bold">{activeEv.confidence}%</span>
              </div>
            </div>
          </div>
        </GlassCard>
      )}
    </div>
  );

  const renderReasoningTab = () => (
    <div className="space-y-5">
      <GlassCard hover={false} className="p-5">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" /> Agentic Reasoning Trace
        </h3>
        <p className="text-xs text-muted-foreground mb-4">
          Step-by-step verification reasoning for: <span className="text-foreground font-semibold">{claim.field}</span>
        </p>

        <div className="relative border-l-2 border-border/60 ml-4">
          {(claim.reasoning || []).map((step, idx) => {
            const colonIdx = step.indexOf(":");
            const stepName = colonIdx > 0 ? step.substring(0, colonIdx).trim() : `Step ${idx + 1}`;
            const stepDetail = colonIdx > 0 ? step.substring(colonIdx + 1).trim() : step;
            const meta = REASONING_STEP_ICONS[stepName] || { icon: <Sparkles className="w-4 h-4" />, color: "text-muted-foreground border-border" };

            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.12, duration: 0.3 }}
                className="relative pl-6 pb-5"
              >
                <div className={`absolute -left-[13px] top-0 w-6 h-6 rounded-full bg-surface-dark border-2 ${meta.color} flex items-center justify-center`}>
                  {meta.icon}
                </div>
                <div className="glass-light rounded-xl p-3.5">
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${meta.color.split(" ")[0]} block mb-1`}>
                    {idx + 1}. {stepName}
                  </span>
                  <p className="text-xs text-muted-foreground leading-relaxed">{stepDetail}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </GlassCard>
    </div>
  );

  const renderComparisonTab = () => (
    <div className="space-y-5">
      <GlassCard hover={false} className="p-5">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-accent" /> Claim Verification Comparison
        </h3>

        <div className="space-y-3">
          <div className="glass-light rounded-xl p-4 border-l-4 border-danger/60">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-danger flex items-center gap-1.5">
                <AlertTriangle className="w-3 h-3" /> Stage 1 · Original AI Claim
              </span>
              <span className="text-[10px] text-muted-foreground">Unverified output</span>
            </div>
            <p className="text-sm font-mono font-semibold text-danger">{claim.originalValue}</p>
          </div>

          <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-muted-foreground rotate-90" /></div>

          <div className="glass-light rounded-xl p-4 border-l-4 border-accent/60">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-accent flex items-center gap-1.5">
                <Eye className="w-3 h-3" /> Stage 2 · Retrieved Evidence
              </span>
              <span className="text-[10px] text-muted-foreground">
                {activeEv ? `Page ${activeEv.pageNumber} · ${activeEv.layoutRegion}` : "No evidence"}
              </span>
            </div>
            <p className="text-sm font-mono text-foreground leading-relaxed">
              {activeEv?.text || "The retrieval engine found no matching region."}
            </p>
          </div>

          <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-muted-foreground rotate-90" /></div>

          <div className={`glass-light rounded-xl p-4 border-l-4 ${
            claim.status === "verified" ? "border-success/60" :
            claim.status === "corrected" ? "border-warning/60" :
            claim.status === "unsupported" ? "border-danger/60" : "border-accent/60"
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5 ${statusColor}`}>
                {statusIcon} Stage 3 · Verified Output
              </span>
              <span className={`text-[10px] font-bold uppercase ${statusColor}`}>{claim.status.replace("_", " ")}</span>
            </div>
            <p className={`text-sm font-mono font-bold ${statusColor}`}>
              {claim.verifiedValue ?? (claim.status === "unsupported" ? "No supporting evidence found" : "Withheld for human review")}
            </p>
          </div>

          <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-muted-foreground rotate-90" /></div>

          <div className="glass-light rounded-xl p-4 border-l-4 border-primary/60">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-primary flex items-center gap-1.5">
                <Sparkles className="w-3 h-3" /> Stage 4 · Explanation
              </span>
              <span className="text-[10px] font-bold text-primary">{claim.trustScore}% trust</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">{claim.reason}</p>
          </div>
        </div>
      </GlassCard>
    </div>
  );

  const renderTimelineTab = () => (
    <div className="space-y-5">
      {claim.timeline && claim.timeline.length > 0 ? (
        <GlassCard hover={false} className="p-5">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-warning" />
            <h3 className="text-sm font-semibold">Verification Timeline</h3>
          </div>
          <div className="relative border-l-2 border-border/60 ml-3 space-y-4 text-xs">
            {claim.timeline.map((evt) => (
              <div key={evt.id} className="relative pl-5">
                <div className="absolute -left-[9px] top-0.5 w-4 h-4 rounded-full bg-surface-dark border-2 border-primary" />
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-mono text-muted-foreground font-semibold">{evt.durationMs} ms</span>
                  <span className="font-bold text-foreground">{evt.title}</span>
                </div>
                <p className="text-muted-foreground leading-relaxed">{evt.detail}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      ) : (
        <GlassCard hover={false} className="p-8 text-center">
          <Clock className="w-8 h-8 mx-auto mb-3 text-muted-foreground/40" />
          <p className="text-xs text-muted-foreground">
            Stage timings are measured per document, not per claim — see the pipeline panel beside the results.
          </p>
        </GlassCard>
      )}
    </div>
  );

  const tabContent: Record<DrawerTab, () => React.ReactNode> = {
    overview: renderOverviewTab,
    evidence: renderEvidenceTab,
    reasoning: renderReasoningTab,
    comparison: renderComparisonTab,
    timeline: renderTimelineTab,
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md flex justify-end"
        onClick={onClose}
      >
        <motion.div
          initial={{ x: "100%" }}
          animate={{ x: 0 }}
          exit={{ x: "100%" }}
          transition={{ type: "spring", damping: 25, stiffness: 200 }}
          className="w-full max-w-2xl h-full glass border-l border-border/60 flex flex-col shadow-2xl overflow-hidden text-foreground"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-5 border-b border-border/50 flex items-center justify-between bg-surface-dark/60">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-widest text-primary">Verification Inspection Drawer</span>
              <h2 className="text-xl font-bold">{claim.field}</h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-xl glass-light hover:bg-surface-light text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close inspection drawer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="px-5 pt-3 pb-0 border-b border-border/40 bg-surface-dark/30">
            <div className="flex items-center gap-1 overflow-x-auto" role="tablist">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  role="tab"
                  aria-selected={activeTab === tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-t-lg transition-all whitespace-nowrap border-b-2 ${
                    activeTab === tab.id
                      ? "text-primary border-primary bg-primary/10"
                      : "text-muted-foreground border-transparent hover:text-foreground hover:bg-surface-light/30"
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                {tabContent[activeTab]()}
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
