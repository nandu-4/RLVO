import { ShieldCheck, AlertTriangle, ShieldAlert, Layers, Shield } from "lucide-react";
import { VerificationSummary } from "@/types/truthlens";
import GlassCard from "./GlassCard";

interface VerificationSummaryCardsProps {
  summary: VerificationSummary;
}

export default function VerificationSummaryCards({ summary }: VerificationSummaryCardsProps) {
  const getRiskBadge = (risk: string) => {
    switch (risk) {
      case "LOW":
        return "bg-success/20 text-success border-success/40";
      case "MEDIUM":
        return "bg-warning/20 text-warning border-warning/40";
      case "HIGH RISK":
      default:
        return "bg-danger/20 text-danger border-danger/40 animate-pulse";
    }
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-6 gap-3.5 mb-8">
      {/* Trust Score & Risk Pill */}
      <GlassCard hover={false} className="col-span-2 p-4 border-l-4 border-l-primary flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
              Overall Trust Score
            </span>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-black border ${getRiskBadge(summary.riskLevel)}`}>
              {summary.riskLevel}
            </span>
          </div>
          <div className="text-3xl font-black gradient-text">{summary.trustScore}%</div>
        </div>
        <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
          <Shield className="w-6 h-6" />
        </div>
      </GlassCard>

      {/* Total Claims */}
      <GlassCard hover={false} className="p-3.5 text-center">
        <div className="text-xl font-bold text-foreground mb-0.5">{summary.totalClaims}</div>
        <div className="text-[11px] text-muted-foreground flex items-center justify-center gap-1">
          <Layers className="w-3 h-3 text-primary" /> Total Claims
        </div>
      </GlassCard>

      {/* Verified */}
      <GlassCard hover={false} className="p-3.5 text-center">
        <div className="text-xl font-bold text-success mb-0.5">{summary.verifiedCount}</div>
        <div className="text-[11px] text-muted-foreground flex items-center justify-center gap-1">
          <ShieldCheck className="w-3 h-3 text-success" /> Verified
        </div>
      </GlassCard>

      {/* Corrected */}
      <GlassCard hover={false} className="p-3.5 text-center">
        <div className="text-xl font-bold text-warning mb-0.5">{summary.correctedCount}</div>
        <div className="text-[11px] text-muted-foreground flex items-center justify-center gap-1">
          <AlertTriangle className="w-3 h-3 text-warning" /> Corrected
        </div>
      </GlassCard>

      {/* Unsupported */}
      <GlassCard hover={false} className="p-3.5 text-center">
        <div className="text-xl font-bold text-danger mb-0.5">{summary.unsupportedCount}</div>
        <div className="text-[11px] text-muted-foreground flex items-center justify-center gap-1">
          <ShieldAlert className="w-3 h-3 text-danger" /> Unsupported
        </div>
      </GlassCard>
    </div>
  );
}
