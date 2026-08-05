import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ShieldAlert,
  Layers,
  ThumbsUp,
  ThumbsDown,
  Edit3,
  Sparkles,
  FileText,
  Gauge,
  Radar,
} from "lucide-react";
import { Claim, ClaimStatus, HumanFeedback } from "@/types/truthlens";
import GlassCard from "./GlassCard";

interface VerificationCardProps {
  claim: Claim;
  isSelected: boolean;
  /** False when the run was not persisted — a decision would have nothing durable to attach to. */
  reviewEnabled: boolean;
  onSelect: () => void;
  onFeedback?: (claimId: string, feedback: HumanFeedback) => void | Promise<void>;
}

export default function VerificationCard({
  claim,
  isSelected,
  reviewEnabled,
  onSelect,
  onFeedback,
}: VerificationCardProps) {
  // Read the decision straight from the claim. Mirroring it into local state meant the card
  // kept showing a decision the server had rejected, and drifted from the drawer.
  const feedback = claim.feedback;

  const getStatusBadge = (status: ClaimStatus) => {
    switch (status) {
      case "verified":
        return {
          label: "Verified",
          icon: <CheckCircle2 className="w-3.5 h-3.5" />,
          bg: "bg-success/15 text-success border-success/30",
        };
      case "corrected":
        return {
          label: "Corrected",
          icon: <AlertTriangle className="w-3.5 h-3.5" />,
          bg: "bg-warning/15 text-warning border-warning/30",
        };
      case "unsupported":
        return {
          label: "Unsupported",
          icon: <ShieldAlert className="w-3.5 h-3.5" />,
          bg: "bg-danger/15 text-danger border-danger/30",
        };
      case "needs_review":
      default:
        return {
          label: "Needs Review",
          icon: <HelpCircle className="w-3.5 h-3.5" />,
          bg: "bg-accent/15 text-accent border-accent/30",
        };
    }
  };

  // Only approve/reject are expressible here. An override needs a corrected value, so it opens
  // the drawer instead of recording an override of nothing, which is what it used to do.
  const handleAction = (e: React.MouseEvent, type: "approved" | "rejected") => {
    e.stopPropagation();
    void onFeedback?.(claim.id, { status: type, timestamp: new Date().toISOString() });
  };

  const badge = getStatusBadge(claim.status);

  return (
    <GlassCard
      onClick={onSelect}
      className={`relative cursor-pointer transition-all duration-300 p-5 ${
        isSelected
          ? "ring-2 ring-primary border-primary/50 bg-primary/10 shadow-lg shadow-primary/10"
          : "hover:border-primary/30 hover:bg-surface-light/40"
      }`}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          {claim.category && (
            <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground block mb-1">
              {claim.category}
            </span>
          )}
          <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
            {claim.field}
          </h3>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${badge.bg}`}>
            {badge.icon}
            {badge.label}
          </div>
          <div className="glass-light px-2.5 py-1 rounded-full text-xs font-bold text-primary border border-primary/20">
            Trust: {claim.trustScore}%
          </div>
          {/* Predicted before verification ran, so it explains rather than restates the outcome. */}
          {claim.hallucinationRisk && claim.hallucinationRisk.level !== "LOW" && (
            <div
              className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-bold uppercase border ${
                claim.hallucinationRisk.level === "HIGH"
                  ? "bg-danger/15 text-danger border-danger/30"
                  : "bg-warning/15 text-warning border-warning/30"
              }`}
              title={claim.hallucinationRisk.reasons[0]}
            >
              <Radar className="w-3 h-3" /> {claim.hallucinationRisk.level} pre-risk
            </div>
          )}
        </div>
      </div>

      {/* Claim Value Transition */}
      <div className="glass-light rounded-xl p-3.5 mb-3.5 space-y-1.5 font-mono text-xs">
        <div className="flex items-center justify-between text-muted-foreground">
          <span className="text-[11px] font-sans font-medium uppercase tracking-wider">
            Original AI Claim
          </span>
          <span className="line-through text-danger/80">{claim.originalValue}</span>
        </div>
        {claim.status === "corrected" && claim.verifiedValue && (
          <div className="flex items-center justify-between pt-1 border-t border-border/40">
            <span className="text-[11px] font-sans font-semibold uppercase tracking-wider text-success flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> Verified Truth
            </span>
            <span className="text-success font-bold">{claim.verifiedValue}</span>
          </div>
        )}
        {claim.status === "verified" && (
          <div className="flex items-center justify-between pt-1 border-t border-border/40">
            <span className="text-[11px] font-sans font-semibold uppercase tracking-wider text-foreground">
              Confirmed Output
            </span>
            <span className="text-foreground font-semibold">{claim.originalValue}</span>
          </div>
        )}
        {claim.status === "unsupported" && (
          <div className="flex items-center justify-between pt-1 border-t border-border/40 text-danger">
            <span className="text-[11px] font-sans font-semibold uppercase tracking-wider">
              Evidence Status
            </span>
            <span className="font-semibold italic">No Evidence Found</span>
          </div>
        )}
      </div>

      {/* Explainable AI (XAI) Reason Snippet */}
      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed mb-4">
        <span className="font-semibold text-foreground">Reason: </span>
        {claim.reason}
      </p>

      {/* Footer Provenance & Human Feedback */}
      <div className="flex items-center justify-between pt-3 border-t border-border/50 text-xs gap-2">
        <div className="flex items-center gap-3 text-muted-foreground min-w-0">
          <span className="flex items-center gap-1 text-[11px] whitespace-nowrap" title="Evidence cited by the verifier out of everything retrieval found">
            <Layers className="w-3.5 h-3.5 text-primary" /> {claim.retrieval?.citedCount ?? 0}/{claim.retrieval?.candidateCount ?? claim.evidence.length} evidence
          </span>
          {claim.evidence[0] && (
            <span className="flex items-center gap-1 text-[11px] whitespace-nowrap">
              <FileText className="w-3.5 h-3.5 text-accent" /> Pg {claim.evidence[0].pageNumber}
            </span>
          )}
          <span
            className="flex items-center gap-1 text-[11px] whitespace-nowrap"
            title={`${claim.confidenceBreakdown.measuredCount} of 5 trust signals were independently measured`}
          >
            <Gauge className="w-3.5 h-3.5 text-secondary" /> {claim.confidenceBreakdown.measuredCount}/5
          </span>
        </div>

        {/* Human Feedback Buttons */}
        <div className="flex items-center gap-1">
          <button
            title={reviewEnabled ? "Approve verification" : "Review requires a stored verification"}
            disabled={!reviewEnabled}
            onClick={(e) => handleAction(e, "approved")}
            className={`p-1.5 rounded-lg border transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
              feedback?.status === "approved"
                ? "bg-success/20 border-success text-success"
                : "hover:bg-surface-light border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button
            title={reviewEnabled ? "Reject verification" : "Review requires a stored verification"}
            disabled={!reviewEnabled}
            onClick={(e) => handleAction(e, "rejected")}
            className={`p-1.5 rounded-lg border transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
              feedback?.status === "rejected"
                ? "bg-danger/20 border-danger text-danger"
                : "hover:bg-surface-light border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
          <button
            title="Override with a corrected value"
            onClick={(e) => {
              e.stopPropagation();
              onSelect();
            }}
            className={`p-1.5 rounded-lg border transition-all ${
              feedback?.status === "overridden"
                ? "bg-primary/20 border-primary text-primary"
                : "hover:bg-surface-light border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <Edit3 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </GlassCard>
  );
}
