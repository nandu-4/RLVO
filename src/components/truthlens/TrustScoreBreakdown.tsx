import { motion } from "framer-motion";
import { Shield, Info, MinusCircle, Lightbulb } from "lucide-react";
import { ConfidenceBreakdown, SignalKey } from "@/types/truthlens";
import GlassCard from "./GlassCard";

interface TrustScoreBreakdownProps {
  breakdown: ConfidenceBreakdown;
  trustScore: number;
}

const SIGNALS: { key: SignalKey; label: string; bar: string; text: string }[] = [
  { key: "ocrAgreement", label: "OCR Agreement", bar: "bg-primary", text: "text-primary" },
  { key: "visionAgreement", label: "Vision Agreement", bar: "bg-accent", text: "text-accent" },
  { key: "layoutAgreement", label: "Layout Agreement", bar: "bg-secondary", text: "text-secondary" },
  { key: "semanticAgreement", label: "Semantic Agreement", bar: "bg-success", text: "text-success" },
  { key: "evidenceStrength", label: "Evidence Strength", bar: "bg-warning", text: "text-warning" },
];

/**
 * Shows not just what each signal scored, but what it was measured from and which signals were
 * excluded. A signal nothing measured is rendered as excluded rather than as a number — the
 * earlier version filled the gaps by copying evidence strength into all four agreement rows,
 * which made five views of one measurement look like independent corroboration.
 */
export default function TrustScoreBreakdown({ breakdown, trustScore }: TrustScoreBreakdownProps) {
  const unmeasured = new Set<SignalKey>(breakdown.unmeasured || []);

  return (
    <GlassCard hover={false} className="p-5">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary" />
          <h3 className="text-sm font-semibold">Trust Score Breakdown</h3>
        </div>
        <div className="text-2xl font-black gradient-text">{trustScore}%</div>
      </div>
      <p className="text-[11px] text-muted-foreground mb-4">
        Weighted mean over the {breakdown.measuredCount} signal{breakdown.measuredCount === 1 ? "" : "s"} that were
        independently measured. Excluded signals do not count as zero and are not substituted.
      </p>

      <div className="space-y-2.5 mb-4">
        {SIGNALS.map((signal) => {
          const value = breakdown[signal.key];
          const excluded = unmeasured.has(signal.key);
          return (
            <div key={signal.key} className={`glass-light rounded-lg p-3 ${excluded ? "opacity-60" : ""}`}>
              <div className="flex justify-between items-center mb-1.5 text-xs">
                <span className="font-medium text-muted-foreground flex items-center gap-1.5">
                  {excluded && <MinusCircle className="w-3 h-3 text-muted-foreground" />}
                  {signal.label}
                </span>
                {excluded ? (
                  <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Not measured</span>
                ) : (
                  <span className={`font-bold ${signal.text}`}>{value}%</span>
                )}
              </div>
              {!excluded && (
                <div className="w-full h-1.5 rounded-full bg-surface-dark overflow-hidden mb-1.5">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.7, ease: "easeOut" }}
                    className={`h-full ${signal.bar}`}
                  />
                </div>
              )}
              <p className="text-[10px] text-muted-foreground leading-relaxed flex items-start gap-1.5">
                <Info className="w-3 h-3 shrink-0 mt-0.5 opacity-60" />
                {breakdown.basis?.[signal.key] || "No basis was recorded for this signal."}
              </p>
            </div>
          );
        })}
      </div>

      {/* The "why does this score exist" narrative the enterprise reviewer actually needs. */}
      {breakdown.why?.length > 0 && (
        <div className="glass-light rounded-lg p-3.5">
          <h4 className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
            <Lightbulb className="w-3.5 h-3.5 text-warning" /> Why this score
          </h4>
          <ul className="space-y-1.5">
            {breakdown.why.map((reason, index) => (
              <li key={index} className="text-[11px] text-muted-foreground leading-relaxed flex gap-2">
                <span className="text-primary font-bold shrink-0">·</span>
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </GlassCard>
  );
}
