import { useState } from "react";
import { Link } from "react-router-dom";
import { Shield, Mail, Loader2, User, ShieldCheck, ArrowRight, FlaskConical } from "lucide-react";
import ParticleBackground from "@/components/truthlens/ParticleBackground";
import MouseGlow from "@/components/truthlens/MouseGlow";
import TruthLensNavbar from "@/components/truthlens/TruthLensNavbar";
import { useAuth } from "@/integrations/auth";

/**
 * Dedicated sign-in page.
 *
 * Two first-class options, matching the task: "Continue with Google" via
 * supabase.auth.signInWithOAuth (wrapped in AuthProvider.signInWithGoogle) and
 * "Continue with Email" via magic-link OTP. The navbar AccountMenu keeps the same two options
 * for quick access from anywhere; this page gives them a proper home and a clear explanation
 * of demo vs workspace mode.
 */
export default function TruthLensLogin() {
  const { configured, loading, user, signInWithGoogle, signInWithEmail } = useAuth();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState<"google" | "email" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [emailSent, setEmailSent] = useState(false);
  const [emailSentTo, setEmailSentTo] = useState("");

  async function continueWithGoogle() {
    setBusy("google");
    setError(null);
    const { error: signInError } = await signInWithGoogle();
    setError(signInError);
    setBusy(null);
  }

  async function continueWithEmail(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim()) return;
    setBusy("email");
    setError(null);
    setEmailSent(false);
    const { error: signInError } = await signInWithEmail(email.trim());
    setError(signInError);
    if (!signInError) {
      setEmailSent(true);
      setEmailSentTo(email.trim());
    }
    setBusy(null);
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col aurora-bg text-foreground">
        <ParticleBackground />
        <MouseGlow />
        <TruthLensNavbar />
        <main id="main" tabIndex={-1} className="relative z-10 flex-1 flex items-center justify-center px-4">
          <div className="glass rounded-2xl p-10 flex items-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="w-5 h-5 text-primary animate-spin" /> Checking session…
          </div>
        </main>
      </div>
    );
  }

  if (!configured) {
    return (
      <div className="min-h-screen flex flex-col aurora-bg text-foreground">
        <ParticleBackground />
        <MouseGlow />
        <TruthLensNavbar />
        <main id="main" tabIndex={-1} className="relative z-10 flex-1 flex items-center justify-center px-4">
          <div className="glass rounded-2xl p-10 text-center max-w-md w-full">
            <div className="w-14 h-14 rounded-2xl bg-warning/15 border border-warning/30 flex items-center justify-center mx-auto mb-5 text-warning">
              <FlaskConical className="w-7 h-7" />
            </div>
            <h1 className="text-lg font-bold">Authentication is not configured</h1>
            <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
              This deployment is running in demo mode. Set <span className="font-mono text-[11px]">VITE_SUPABASE_URL</span> and{" "}
              <span className="font-mono text-[11px]">VITE_SUPABASE_ANON_KEY</span> to enable Google and email sign-in.
            </p>
            <Link to="/verify" className="btn-primary text-xs py-2.5 px-5 inline-block mt-6">
              Continue in demo mode <ArrowRight className="w-3.5 h-3.5 inline ml-1" />
            </Link>
          </div>
        </main>
      </div>
    );
  }

  // Already signed in — show the account and let the user continue rather than sending them
  // through the sign-in flow again.
  if (user) {
    return (
      <div className="min-h-screen flex flex-col aurora-bg text-foreground">
        <ParticleBackground />
        <MouseGlow />
        <TruthLensNavbar />
        <main id="main" tabIndex={-1} className="relative z-10 flex-1 flex items-center justify-center px-4">
          <div className="glass rounded-2xl p-10 text-center max-w-md w-full">
            {user.avatarUrl ? (
              <img src={user.avatarUrl} alt="" className="w-16 h-16 rounded-2xl object-cover mx-auto mb-5 border border-border" />
            ) : (
              <div className="w-16 h-16 rounded-2xl bg-primary/15 text-primary flex items-center justify-center mx-auto mb-5">
                <User className="w-8 h-8" />
              </div>
            )}
            <h1 className="text-xl font-bold">Signed in</h1>
            <p className="text-sm text-muted-foreground mt-1 break-anywhere">{user.email}</p>
            <p className="text-xs text-success font-semibold mt-3 flex items-center justify-center gap-1.5">
              <ShieldCheck className="w-4 h-4" /> {user.name} · Workspace mode
            </p>
            <Link to="/verify" className="btn-primary text-sm py-2.5 px-6 inline-block mt-6">
              Go to Verification <ArrowRight className="w-4 h-4 inline ml-1" />
            </Link>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col aurora-bg text-foreground">
      <ParticleBackground />
      <MouseGlow />
      <TruthLensNavbar />
      <main id="main" tabIndex={-1} className="relative z-10 flex-1 flex items-center justify-center px-4 py-28">
        <div className="glass rounded-2xl p-8 md:p-10 max-w-md w-full">
          <div className="flex items-center gap-2.5 mb-6">
            <Shield className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-bold">
              Sign in to <span className="gradient-text">TruthLens</span>
            </h1>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed mb-8">
            Sign in to save verifications, open the review queue, and record decisions to your workspace. Guests can always
            verify in demo mode without an account.
          </p>

          <div className="flex flex-col gap-3">
            <button
              type="button"
              onClick={() => void continueWithGoogle()}
              disabled={busy !== null}
              className="w-full py-3 flex items-center justify-center gap-2.5 disabled:opacity-60 relative rounded-xl bg-[#2d3748] text-white font-semibold text-sm transition-all duration-300 hover:bg-[#374151] hover:translate-y-[-1px]"
            >
              {busy === "google" ? <Loader2 className="w-4 h-4 animate-spin" /> : (
                <svg className="w-4 h-4" viewBox="0 0 24 24" aria-hidden="true">
                  <path fill="#4285F4" d="M23.49 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47a5.53 5.53 0 0 1-2.4 3.63v3h3.87c2.27-2.09 3.55-5.17 3.55-8.87z" />
                  <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.87-3c-1.08.72-2.45 1.16-4.06 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.09A11.99 11.99 0 0 0 12 24z" />
                  <path fill="#FBBC05" d="M5.27 14.29a7.18 7.18 0 0 1 0-4.58V6.62H1.29a12 12 0 0 0 0 10.76l3.98-3.09z" />
                  <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.69 1.29 6.62l3.98 3.09C6.22 6.86 8.87 4.75 12 4.75z" />
                </svg>
              )}
              Continue with Google
            </button>

            <div className="flex items-center gap-3" aria-hidden="true">
              <span className="h-px flex-1 bg-border" />
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground">or</span>
              <span className="h-px flex-1 bg-border" />
            </div>

            <form className="flex flex-col gap-3" onSubmit={(event) => void continueWithEmail(event)}>
              <label htmlFor="login-email" className="sr-only">Email address</label>
              <input
                id="login-email"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                required
                disabled={busy !== null}
                className="h-11 rounded-xl border border-border bg-background px-3.5 text-sm focus:outline-none focus:border-primary disabled:opacity-60"
              />
              <button type="submit" disabled={busy !== null} className="btn-primary w-full py-3 flex items-center justify-center gap-2.5 disabled:opacity-60">
                {busy === "email" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Mail className="w-4 h-4" />}
                Continue with Email
              </button>
            </form>

            {emailSent && (
              <p className="text-xs text-success leading-relaxed">
                A secure sign-in link was sent to <span className="font-semibold break-anywhere">{emailSentTo}</span>. Check your
                inbox and return to this page.
              </p>
            )}
            {error && <p className="text-xs text-danger leading-relaxed">{error}</p>}
          </div>

          <p className="text-[11px] text-muted-foreground mt-8 text-center leading-relaxed">
            No account needed to try the engine — verification works in demo mode from the{" "}
            <Link to="/verify" className="text-primary hover:underline">verification page</Link>.
          </p>
        </div>
      </main>
    </div>
  );
}
