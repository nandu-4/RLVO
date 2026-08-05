import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Download, FileSpreadsheet, History, FileText } from "lucide-react";
import { VerificationResult } from "@/types/truthlens";
import { exportCsv, exportJson, openPrintableReport, reportRef } from "@/lib/verificationReport";

interface AuditTrailDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  result: VerificationResult | null;
}

export default function AuditTrailDrawer({ isOpen, onClose, result }: AuditTrailDrawerProps) {
  const [exportError, setExportError] = useState<string | null>(null);
  if (!isOpen || !result) return null;

  const documentRef = reportRef(result);

  const handlePrint = () => {
    setExportError(openPrintableReport(result) ? null : "The report window was blocked. Allow pop-ups for this site and try again.");
  };

  const entries = result.claims.map((claim) => ({
    id: `at-${claim.id}`,
    claimId: claim.id,
    field: claim.field,
    originalValue: claim.originalValue,
    finalValue: claim.verifiedValue ?? claim.originalValue,
    changed: Boolean(claim.verifiedValue && claim.verifiedValue !== claim.originalValue),
    status: claim.status,
    trustScore: claim.trustScore,
    signals: `${claim.confidenceBreakdown.measuredCount}/5`,
    risk: claim.hallucinationRisk?.level ?? "—",
    reviewer: claim.feedback ? "Human reviewer" : "Automated engine",
    timestamp: claim.feedback?.timestamp || result.createdAt,
  }));

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md flex items-center justify-center p-4 md:p-8"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="w-full max-w-5xl h-[85vh] glass border border-border/60 rounded-2xl flex flex-col shadow-2xl overflow-hidden text-foreground"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-border/50 flex items-center justify-between gap-4 bg-surface-dark/60">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
                <History className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <h2 className="text-xl font-bold">Compliance Audit Trail</h2>
                <p className="text-xs text-muted-foreground truncate">
                  {result.fileName} · {documentRef} · {result.documentType}
                  {!result.documentId && <span className="text-warning font-semibold"> · not stored</span>}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button onClick={() => exportJson(result)} className="btn-secondary py-2 px-3 text-xs flex items-center gap-1.5">
                <Download className="w-3.5 h-3.5 text-primary" /> JSON
              </button>
              <button onClick={() => exportCsv(result)} className="btn-secondary py-2 px-3 text-xs flex items-center gap-1.5">
                <FileSpreadsheet className="w-3.5 h-3.5 text-primary" /> CSV
              </button>
              <button onClick={handlePrint} className="btn-primary py-2 px-3 text-xs flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" /> Full report (PDF)
              </button>
              <button onClick={onClose} className="p-2 rounded-xl glass-light hover:bg-surface-light text-muted-foreground" aria-label="Close audit trail">
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {exportError && <div className="px-6 pt-4 text-xs text-danger">{exportError}</div>}

          <div className="flex-1 overflow-auto p-6">
            <div className="glass rounded-xl overflow-hidden border border-border/50">
              <table className="w-full text-xs text-left">
                <thead className="bg-surface-dark/80 text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border/50">
                  <tr>
                    <th className="p-3.5">Field</th>
                    <th className="p-3.5">Original AI Claim</th>
                    <th className="p-3.5">Final Value</th>
                    <th className="p-3.5">Status</th>
                    <th className="p-3.5 text-center">Trust</th>
                    <th className="p-3.5 text-center">Signals</th>
                    <th className="p-3.5 text-center">Pre-risk</th>
                    <th className="p-3.5">Decided by</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {entries.map((entry) => (
                    <tr key={entry.id} className="hover:bg-surface-light/30 transition-colors">
                      <td className="p-3.5 font-semibold text-foreground">{entry.field}</td>
                      {/* Only strike a value that was actually superseded — striking every original
                          made a correctly-verified claim read as if it had been corrected. */}
                      <td className={`p-3.5 font-mono ${entry.changed ? "text-danger line-through" : "text-muted-foreground"}`}>
                        {entry.originalValue}
                      </td>
                      <td className={`p-3.5 font-mono font-bold ${entry.changed ? "text-success" : "text-foreground"}`}>
                        {entry.finalValue}
                      </td>
                      <td className="p-3.5">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                            entry.status === "verified"
                              ? "bg-success/20 text-success"
                              : entry.status === "corrected"
                              ? "bg-warning/20 text-warning"
                              : entry.status === "unsupported"
                              ? "bg-danger/20 text-danger"
                              : "bg-accent/20 text-accent"
                          }`}
                        >
                          {entry.status.replace("_", " ")}
                        </span>
                      </td>
                      <td className="p-3.5 text-center font-bold text-primary">{entry.trustScore}%</td>
                      <td className="p-3.5 text-center font-mono text-muted-foreground">{entry.signals}</td>
                      <td className="p-3.5 text-center font-mono text-muted-foreground">{entry.risk}</td>
                      <td className="p-3.5 text-muted-foreground">
                        {entry.reviewer}
                        <span className="block text-[10px] opacity-70">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
