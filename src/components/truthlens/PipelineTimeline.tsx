import { motion } from "framer-motion";
import { CheckCircle2, AlertTriangle, XCircle, Info, Clock } from "lucide-react";
import { VerificationTimelineEvent } from "@/types/truthlens";
import GlassCard from "./GlassCard";

interface PipelineTimelineProps {
  events: VerificationTimelineEvent[];
  totalMs: number;
}

const STATUS_STYLE: Record<string, { icon: React.ReactNode; ring: string; text: string }> = {
  success: { icon: <CheckCircle2 className="w-3.5 h-3.5" />, ring: "border-success", text: "text-success" },
  warning: { icon: <AlertTriangle className="w-3.5 h-3.5" />, ring: "border-warning", text: "text-warning" },
  danger: { icon: <XCircle className="w-3.5 h-3.5" />, ring: "border-danger", text: "text-danger" },
  info: { icon: <Info className="w-3.5 h-3.5" />, ring: "border-primary", text: "text-primary" },
};

/**
 * Renders the stages the server actually executed, with their measured durations.
 * There is no client-side animation of imaginary steps here: if a stage is absent from the
 * response it did not run, and its absence is the honest thing to show.
 */
export default function PipelineTimeline({ events, totalMs }: PipelineTimelineProps) {
  if (events.length === 0) return null;
  const slowest = Math.max(...events.map((event) => event.durationMs), 1);

  return (
    <GlassCard hover={false} className="p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Clock className="w-4 h-4 text-primary" /> Measured pipeline stages
        </h3>
        <span className="text-xs text-muted-foreground font-mono">{totalMs.toLocaleString()} ms total</span>
      </div>

      <div className="space-y-2.5">
        {events.map((event, index) => {
          const style = STATUS_STYLE[event.status] || STATUS_STYLE.info;
          return (
            <motion.div
              key={`${event.id}-${index}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.06, duration: 0.25 }}
              className="glass-light rounded-xl p-3"
            >
              <div className="flex items-center justify-between gap-3 mb-1.5">
                <span className={`text-xs font-semibold flex items-center gap-1.5 ${style.text}`}>
                  {style.icon}
                  {event.title}
                </span>
                <span className="text-[11px] font-mono text-muted-foreground shrink-0">
                  {event.durationMs.toLocaleString()} ms
                </span>
              </div>
              <div className="h-1 rounded-full bg-surface-dark overflow-hidden mb-1.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(2, (event.durationMs / slowest) * 100)}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className={`h-full ${style.text.replace("text-", "bg-")}`}
                />
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{event.detail}</p>
            </motion.div>
          );
        })}
      </div>
    </GlassCard>
  );
}
