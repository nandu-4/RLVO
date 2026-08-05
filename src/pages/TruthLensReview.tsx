import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ClipboardList, Loader2, RefreshCw, UserPlus, UserMinus, CheckCircle2, Radar, Database, FileText } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import { invokeAi } from "@/integrations/aiClient";

interface Task {
  id: string;
  status: "open" | "assigned" | "resolved";
  assignedName: string | null;
  createdAt: string;
  resolvedAt: string | null;
  documentId: string;
  documentName: string;
  documentType: string;
  claimId: string;
  field: string;
  category: string | null;
  originalValue: string;
  verifiedValue: string | null;
  claimStatus: string;
  trustScore: number;
  reason: string;
  risk: string | null;
  evidenceRetrieved: number;
  evidenceCited: number;
}

interface Queue {
  available: boolean;
  reason?: string;
  counts?: { open: number; assigned: number; resolved: number };
  tasks: Task[];
}

const REVIEWER_KEY = "truthlens.reviewer_name";

/**
 * Cross-document review queue.
 *
 * `needs_review` used to be a dead end: tasks were written on every verification and never
 * surfaced. This closes the chain — Needs Review → Assign → Comment → Decide → Audit — by giving
 * the first two steps a home. The decision itself still happens in the claim drawer, where the
 * evidence is.
 */
export default function TruthLensReview() {
  const [queue, setQueue] = useState<Queue | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"open" | "assigned" | "resolved" | "all">("open");
  const [reviewer, setReviewer] = useState("");
  const [busyTask, setBusyTask] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setReviewer(window.localStorage.getItem(REVIEWER_KEY) ?? "");
    } catch {
      /* storage unavailable */
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    const { data, error } = await invokeAi<Queue>("review-queue", { action: "list" });
    if (error) setError(error.message);
    else {
      setQueue(data);
      setError(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const saveReviewer = (name: string) => {
    setReviewer(name);
    try {
      window.localStorage.setItem(REVIEWER_KEY, name);
    } catch {
      /* storage unavailable */
    }
  };

  const assign = async (task: Task, action: "assign" | "unassign") => {
    if (action === "assign" && !reviewer.trim()) {
      setError("Enter your reviewer name first — it is recorded on the audit trail.");
      return;
    }
    setBusyTask(task.id);
    const { error } = await invokeAi("review-queue", { action, taskId: task.id, assignedName: reviewer.trim() });
    if (error) setError(error.message);
    else {
      setError(null);
      await load();
    }
    setBusyTask(null);
  };

  const visible = (queue?.tasks ?? []).filter((task) => filter === "all" || task.status === filter);

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">
                Human <span className="gradient-text">Review Queue</span>
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Claims the engine would not decide automatically, across every document in this workspace.
              </p>
            </div>
            <button onClick={load} disabled={loading} className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 disabled:opacity-50">
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Refresh
            </button>
          </div>

          {error && <GlassCard hover={false} className="p-3.5 mb-5 border-danger/40"><p className="text-xs text-danger">{error}</p></GlassCard>}

          {loading && !queue && (
            <GlassCard hover={false} className="p-12 text-center"><Loader2 className="w-8 h-8 text-primary mx-auto animate-spin" /></GlassCard>
          )}

          {queue && !queue.available && (
            <GlassCard hover={false} className="p-10 text-center max-w-3xl mx-auto">
              <Database className="w-10 h-10 text-primary mx-auto mb-4" />
              <h2 className="text-lg font-bold">The review queue needs a stored workspace</h2>
              <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{queue.reason}</p>
            </GlassCard>
          )}

          {queue?.available && (
            <>
              <GlassCard hover={false} className="p-4 mb-6">
                <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                  <div className="flex-1">
                    <label className="block text-[10px] uppercase font-bold tracking-wider text-muted-foreground mb-1.5">
                      Your reviewer name
                    </label>
                    <input
                      value={reviewer}
                      onChange={(event) => saveReviewer(event.target.value)}
                      placeholder="e.g. Nandini"
                      className="glass-light px-3 py-2 rounded-lg text-xs border border-border bg-transparent w-full max-w-xs focus:outline-none focus:border-primary"
                    />
                    {/* Naming the limitation where the name is entered, not in a footnote. */}
                    <p className="text-[10px] text-muted-foreground mt-1.5 leading-relaxed">
                      Recorded on the audit trail. With no accounts this is self-declared — it captures who
                      <em> said</em> they made a decision, not a verified identity.
                    </p>
                  </div>

                  <div className="flex items-center gap-1.5 flex-wrap">
                    {(["open", "assigned", "resolved", "all"] as const).map((option) => (
                      <button
                        key={option}
                        onClick={() => setFilter(option)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all capitalize ${
                          filter === option
                            ? "bg-primary text-primary-foreground border-primary"
                            : "glass-light border-border text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {option}
                        {option !== "all" && queue.counts && (
                          <span className="ml-1.5 opacity-70">{queue.counts[option]}</span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </GlassCard>

              {visible.length === 0 ? (
                <GlassCard hover={false} className="p-10 text-center">
                  <CheckCircle2 className="w-10 h-10 text-success mx-auto mb-4" />
                  <h2 className="text-lg font-bold">Nothing {filter === "all" ? "in the queue" : `marked ${filter}`}</h2>
                  <p className="text-sm text-muted-foreground mt-2">
                    Claims land here when the engine cannot justify an automatic decision from retrieved evidence.
                  </p>
                  <Link to="/verify" className="btn-secondary text-xs py-2 px-4 inline-block mt-5">Verify a document →</Link>
                </GlassCard>
              ) : (
                <div className="space-y-3">
                  {visible.map((task) => (
                    <GlassCard key={task.id} hover={false} className="p-4">
                      <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-1.5">
                            <span className="text-sm font-bold text-foreground">{task.field}</span>
                            {task.category && (
                              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">{task.category}</span>
                            )}
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-accent/15 text-accent border border-accent/30">
                              {task.claimStatus.replace("_", " ")}
                            </span>
                            {task.risk && task.risk !== "LOW" && (
                              <span
                                className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border flex items-center gap-1 ${
                                  task.risk === "HIGH" ? "bg-danger/15 text-danger border-danger/30" : "bg-warning/15 text-warning border-warning/30"
                                }`}
                              >
                                <Radar className="w-3 h-3" /> {task.risk} pre-risk
                              </span>
                            )}
                          </div>

                          <p className="text-xs font-mono text-muted-foreground mb-1.5">
                            <span className="text-danger">{task.originalValue}</span>
                            {task.verifiedValue && <span className="text-success"> → {task.verifiedValue}</span>}
                          </p>
                          <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">{task.reason}</p>

                          <div className="flex items-center gap-3 text-[10px] text-muted-foreground flex-wrap">
                            <span className="flex items-center gap-1">
                              <FileText className="w-3 h-3 text-primary" /> {task.documentName} · {task.documentType}
                            </span>
                            <span>Trust {task.trustScore}%</span>
                            <span>
                              Evidence {task.evidenceCited}/{task.evidenceRetrieved} cited
                            </span>
                            <span>{new Date(task.createdAt).toLocaleString()}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {task.status === "resolved" ? (
                            <span className="text-xs text-success font-semibold flex items-center gap-1.5">
                              <CheckCircle2 className="w-4 h-4" /> Resolved
                              {task.assignedName && <span className="text-muted-foreground font-normal">by {task.assignedName}</span>}
                            </span>
                          ) : (
                            <>
                              {task.assignedName && (
                                <span className="text-xs text-primary font-semibold">Assigned to {task.assignedName}</span>
                              )}
                              <button
                                onClick={() => assign(task, task.status === "assigned" ? "unassign" : "assign")}
                                disabled={busyTask === task.id}
                                className="btn-secondary text-xs py-2 px-3 flex items-center gap-1.5 disabled:opacity-50"
                              >
                                {busyTask === task.id ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : task.status === "assigned" ? (
                                  <UserMinus className="w-3.5 h-3.5 text-warning" />
                                ) : (
                                  <UserPlus className="w-3.5 h-3.5 text-primary" />
                                )}
                                {task.status === "assigned" ? "Unassign" : "Assign to me"}
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              )}

              <GlassCard hover={false} className="p-4 mt-6">
                <p className="text-[11px] text-muted-foreground leading-relaxed flex items-start gap-2">
                  <ClipboardList className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  Assignment happens here; the decision itself happens in the claim drawer on the verification result,
                  where the evidence, the trust breakdown and the override field are. Approving, rejecting or overriding
                  a claim resolves its task automatically and writes to the audit trail.
                </p>
              </GlassCard>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
