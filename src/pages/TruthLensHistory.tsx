import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { History, Loader2, Trash2, HardDrive, Cloud, AlertTriangle } from "lucide-react";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import { invokeAi } from "@/integrations/aiClient";
import { SignInRequired } from "@/components/truthlens/AccountMenu";

type Session = {
  id: string;
  createdAt: string;
  fileName: string;
  documentType: string;
  provider: string;
  model: string;
  trustScore: number;
  verificationMode?: string;
  totalClaims?: number;
};

type Storage = {
  driver: "supabase" | "local" | "none";
  available: boolean;
  signInRequired?: boolean;
  durability?: "durable" | "ephemeral";
};

/**
 * Stored verification sessions.
 *
 * Gating this on a signed-in user stopped being correct once storage stopped requiring one: a
 * deployment with no Supabase project stores locally and has real history to show, and this page
 * hid it behind a sign-in prompt that could never be satisfied. The server reports which driver
 * is live and whether signing in would change anything; the page renders that rather than guessing.
 */
export default function TruthLensHistory() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [storage, setStorage] = useState<Storage | null>(null);
  const [reason, setReason] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    void invokeAi<{ sessions: Session[]; storage: Storage; reason?: string }>("history", {}).then(({ data, error: err }) => {
      setSessions(data?.sessions ?? []);
      setStorage(data?.storage ?? null);
      setReason(data?.reason ?? null);
      setError(err?.message ?? null);
      setLoading(false);
    });
  }, []);

  useEffect(load, [load]);

  const remove = async (id: string) => {
    await invokeAi("history", { id, action: "delete" });
    setSessions((current) => current.filter((s) => s.id !== id));
  };

  return (
    <div className="min-h-screen aurora-bg text-foreground">
      <TruthLensNavbar />
      <main className="max-w-5xl mx-auto pt-28 px-4 pb-16">
        <div className="flex items-center gap-3 mb-3">
          <History className="text-primary" />
          <div>
            <h1 className="text-3xl font-bold">Verification History</h1>
            <p className="text-sm text-muted-foreground">Replay any saved session — no AI call, no quota spent.</p>
          </div>
        </div>

        {storage?.available && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mb-6">
            {storage.driver === "supabase" ? <Cloud className="w-3.5 h-3.5" /> : <HardDrive className="w-3.5 h-3.5" />}
            <span>
              {storage.driver === "supabase"
                ? "Stored in your account"
                : "Stored on this deployment, scoped to this browser"}
            </span>
            {storage.durability === "ephemeral" && (
              <span className="flex items-center gap-1 text-warning">
                <AlertTriangle className="w-3.5 h-3.5" />
                Temporary storage — set TRUTHLENS_DATA_DIR or configure Supabase to keep sessions permanently
              </span>
            )}
          </div>
        )}

        {loading ? (
          <Loader2 className="animate-spin text-primary" />
        ) : error ? (
          <p className="text-danger">{error}</p>
        ) : storage?.signInRequired ? (
          <SignInRequired feature="Verification history" />
        ) : !storage?.available ? (
          <p className="text-muted-foreground max-w-2xl">{reason}</p>
        ) : sessions.length === 0 ? (
          <p className="text-muted-foreground">No stored verifications yet. Verify a document and it will appear here.</p>
        ) : (
          <div className="grid gap-3">
            {sessions.map((session) => (
              <div
                key={session.id}
                className="glass rounded-xl p-5 border border-border hover:border-primary/50 transition-colors flex items-center gap-4"
              >
                <button onClick={() => navigate(`/verify?replay=${session.id}`)} className="flex-1 text-left">
                  <div className="flex justify-between gap-4">
                    <div>
                      <p className="font-bold">{session.fileName}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {session.documentType} · {session.provider} · {session.model}
                        {session.totalClaims ? ` · ${session.totalClaims} claim(s)` : ""}
                      </p>
                    </div>
                    <div className="text-right text-xs shrink-0">
                      <p className="text-success font-bold">Trust {session.trustScore}%</p>
                      <p className="text-muted-foreground">{session.verificationMode || "cross-check"}</p>
                      <p className="text-muted-foreground">{new Date(session.createdAt).toLocaleString()}</p>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => remove(session.id)}
                  aria-label={`Delete stored verification of ${session.fileName}`}
                  className="text-muted-foreground hover:text-danger p-2 rounded-lg shrink-0"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
