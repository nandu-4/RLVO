import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart3, Database, ShieldCheck, Clock, Loader2, RefreshCw, AlertTriangle, Gauge, Radar, TrendingUp } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import GlassCard from "@/components/truthlens/GlassCard";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import { BarRows, StatTile, TrendChart, type TrendPoint } from "@/components/truthlens/charts";
import { invokeAi } from "@/integrations/aiClient";
import { useAuth } from "@/integrations/auth";
import { SignInRequired } from "@/components/truthlens/AccountMenu";

interface Analytics {
  available: boolean;
  hasData: boolean;
  reason?: string;
  windowDays?: number;
  account?: { name: string; email: string };
  totals?: { documents: number; claims: number; verified: number; corrected: number; unsupported: number; needsReview: number; reviewDecisions: number };
  rates?: {
    averageTrustScore: number;
    correctionRate: number;
    unsupportedRate: number;
    needsReviewRate: number;
    verificationSuccessRate: number;
    averageSignalsMeasured: number;
    evidenceCitationRate: number;
    averageDocumentLegibility: number;
  };
  mostHallucinatedFields?: Array<{ label: string; count: number }>;
  topErrorCategories?: Array<{ label: string; count: number }>;
  topDocumentTypes?: Array<{ label: string; count: number }>;
  modelComparison?: Array<{ label: string; documents: number; averageTrustScore: number; needsReviewRate: number }>;
  riskDistribution?: { low: number; medium: number; high: number };
  trend?: TrendPoint[];
  recentActivity?: Array<{ documentId: string; documentType: string; trustScore: number; riskLevel: string; totalClaims: number; needsReview: number; createdAt: string }>;
}

/**
 * Every figure here is computed from verifications this workspace actually ran. An empty
 * workspace shows an empty state — there is no sample data anywhere in this page.
 */
export default function TruthLensDashboard() {
  const [data, setData] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { user, loading: authLoading } = useAuth();

  const load = async () => {
    setLoading(true);
    const { data, error } = await invokeAi<Analytics>("analytics", {});
    if (error) setError(error.message);
    else {
      setData(data);
      setError(null);
    }
    setLoading(false);
  };

  useEffect(() => {
    // History belongs to an account; there is nothing to fetch for a guest.
    if (user) void load();
    else setLoading(false);
  }, [user]);

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 pt-28 pb-12 px-4 md:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">
                Enterprise AI <span className="gradient-text">Trust Dashboard</span>
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                {data?.hasData
                  ? `Last ${data.windowDays} days · ${data.account?.name ?? "your account"}`
                  : "Verification analytics for your account."}
              </p>
            </div>
            {user && <button onClick={load} disabled={loading} className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 disabled:opacity-50">
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Refresh
            </button>}
          </div>

          {error && <GlassCard hover={false} className="p-5 mb-6 border-danger/40"><p className="text-xs text-danger">{error}</p></GlassCard>}

          {!authLoading && !user && <SignInRequired feature="The dashboard" />}

          {user && loading && !data && (
            <GlassCard hover={false} className="p-12 text-center">
              <Loader2 className="w-8 h-8 text-primary mx-auto animate-spin" />
            </GlassCard>
          )}

          {user && data && !data.available && (
            <GlassCard hover={false} className="p-10 text-center max-w-3xl mx-auto">
              <Database className="w-10 h-10 text-primary mx-auto mb-4" />
              <h2 className="text-lg font-bold">Analytics need a stored workspace</h2>
              <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{data.reason}</p>
            </GlassCard>
          )}

          {user && data?.available && !data.hasData && (
            <GlassCard hover={false} className="p-10 text-center max-w-3xl mx-auto">
              <BarChart3 className="w-10 h-10 text-primary mx-auto mb-4" />
              <h2 className="text-lg font-bold">No verifications in this workspace yet</h2>
              <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                TruthLens deliberately shows no sample metrics. Verify a document and this dashboard fills with
                measurements from your own runs.
              </p>
              <Link to="/verify" className="btn-primary text-xs py-2 px-4 inline-block mt-5 relative z-10">
                Verify a document →
              </Link>
            </GlassCard>
          )}

          {user && data?.hasData && data.totals && data.rates && (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <StatTile label="Documents verified" value={data.totals.documents} icon={<Database className="w-4 h-4" />} hint={`${data.totals.claims} claims checked`} />
                <StatTile label="Average trust score" value={`${data.rates.averageTrustScore}%`} tone="gradient-text" icon={<ShieldCheck className="w-4 h-4" />} hint={`${data.rates.verificationSuccessRate}% of runs needed no human review`} />
                <StatTile label="Correction rate" value={`${data.rates.correctionRate}%`} tone="text-warning" icon={<AlertTriangle className="w-4 h-4" />} hint={`${data.totals.corrected} claims contradicted by evidence`} />
                <StatTile label="Unsupported rate" value={`${data.rates.unsupportedRate}%`} tone="text-danger" icon={<Radar className="w-4 h-4" />} hint={`${data.totals.unsupported} claims the document did not support`} />
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <StatTile label="Needs review" value={`${data.rates.needsReviewRate}%`} tone="text-accent" hint={`${data.totals.needsReview} claims awaiting a human`} />
                <StatTile label="Signals measured" value={`${data.rates.averageSignalsMeasured}/5`} icon={<Gauge className="w-4 h-4" />} hint="Average independently measured trust signals per claim" />
                <StatTile label="Evidence citation" value={`${data.rates.evidenceCitationRate}%`} hint="Share of retrieved candidates the verifier relied on" />
                <StatTile label="Human decisions" value={data.totals.reviewDecisions} icon={<Clock className="w-4 h-4" />} hint="Approvals, rejections and overrides recorded" />
              </div>

              <GlassCard hover={false} className="p-5 mb-6">
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-primary" /> Trust trend
                </h2>
                <TrendChart points={data.trend || []} />
              </GlassCard>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <GlassCard hover={false} className="p-5">
                  <h2 className="text-sm font-semibold mb-1">Most hallucinated fields</h2>
                  <p className="text-[11px] text-muted-foreground mb-4">Fields that most often failed to verify cleanly.</p>
                  <BarRows
                    rows={(data.mostHallucinatedFields || []).map((row) => ({ label: row.label, value: row.count, tone: "bg-warning" }))}
                    emptyLabel="Every claim verified cleanly in this window."
                  />
                </GlassCard>

                <GlassCard hover={false} className="p-5">
                  <h2 className="text-sm font-semibold mb-1">Top error categories</h2>
                  <p className="text-[11px] text-muted-foreground mb-4">Why claims did not pass as stated.</p>
                  <BarRows
                    rows={(data.topErrorCategories || []).map((row, index) => ({
                      label: row.label,
                      value: row.count,
                      tone: ["bg-warning", "bg-danger", "bg-accent"][index] || "bg-primary",
                    }))}
                    emptyLabel="No failed claims in this window."
                  />
                </GlassCard>

                <GlassCard hover={false} className="p-5">
                  <h2 className="text-sm font-semibold mb-1">Top document types</h2>
                  <p className="text-[11px] text-muted-foreground mb-4">Classified from content, not from a preset list.</p>
                  <BarRows rows={(data.topDocumentTypes || []).map((row) => ({ label: row.label, value: row.count }))} />
                </GlassCard>

                <GlassCard hover={false} className="p-5">
                  <h2 className="text-sm font-semibold mb-1">Pre-verification risk distribution</h2>
                  <p className="text-[11px] text-muted-foreground mb-4">Predicted before the verifier ran, from legibility and retrieval strength.</p>
                  <BarRows
                    rows={[
                      { label: "Low risk", value: data.riskDistribution?.low ?? 0, tone: "bg-success" },
                      { label: "Medium risk", value: data.riskDistribution?.medium ?? 0, tone: "bg-warning" },
                      { label: "High risk", value: data.riskDistribution?.high ?? 0, tone: "bg-danger" },
                    ]}
                  />
                </GlassCard>
              </div>

              {(data.modelComparison?.length ?? 0) > 0 && (
                <GlassCard hover={false} className="p-5 mb-6">
                  <h2 className="text-sm font-semibold mb-1">Model comparison</h2>
                  <p className="text-[11px] text-muted-foreground mb-4">
                    Across models actually used in this workspace. For a controlled head-to-head on one document, use{" "}
                    <Link to="/benchmark" className="text-primary hover:underline">Benchmark</Link>.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border/50">
                        <tr>
                          <th className="text-left py-2">Model</th>
                          <th className="text-right py-2">Documents</th>
                          <th className="text-right py-2">Avg trust</th>
                          <th className="text-right py-2">Needs review</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/40">
                        {data.modelComparison?.map((row) => (
                          <tr key={row.label}>
                            <td className="py-2.5 font-mono text-foreground">{row.label}</td>
                            <td className="py-2.5 text-right tabular-nums">{row.documents}</td>
                            <td className="py-2.5 text-right tabular-nums font-bold text-primary">{row.averageTrustScore}%</td>
                            <td className="py-2.5 text-right tabular-nums text-accent">{row.needsReviewRate}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>
              )}

              <GlassCard hover={false} className="p-5">
                <h2 className="text-sm font-semibold mb-4">Recent activity</h2>
                <div className="space-y-2">
                  {(data.recentActivity || []).map((entry) => (
                    <div key={entry.documentId} className="glass-light rounded-lg p-3 flex items-center justify-between gap-3 text-xs">
                      <div className="min-w-0">
                        <span className="font-semibold text-foreground">{entry.documentType}</span>
                        <span className="text-muted-foreground ml-2">{entry.totalClaims} claims</span>
                        {entry.needsReview > 0 && <span className="text-accent ml-2">{entry.needsReview} need review</span>}
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="font-bold text-primary tabular-nums">{entry.trustScore}%</span>
                        <span className="text-muted-foreground">{new Date(entry.createdAt).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
