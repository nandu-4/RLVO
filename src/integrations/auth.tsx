import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Authentication and the two operating modes.
 *
 * Guest    → Demo mode.      Verification works. Nothing is stored. Dashboard, review queue and
 *                            audit are unavailable, and the UI says so rather than failing later.
 * Signed in → Workspace mode. Persistence, dashboard, review queue and audit trail all enabled,
 *                            scoped to the authenticated user.
 *
 * This replaces the previous environment-variable mode switch, which decided capability from
 * deployment config rather than from who was actually using the app — so every visitor shared one
 * anonymous workspace and the audit trail could only record self-declared names.
 *
 * Auth is optional at build time: with no Supabase keys the provider reports `configured: false`
 * and the whole app runs in demo mode. That keeps the project cloneable without an account.
 */

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/** Publishable anon key only — never a service-role key, which must not reach the browser. */
export const supabase: SupabaseClient | null =
  SUPABASE_URL && SUPABASE_ANON_KEY
    ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
      })
    : null;

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  avatarUrl: string | null;
}

interface AuthState {
  /** False when this deployment has no Supabase auth configured; the app stays in demo mode. */
  configured: boolean;
  loading: boolean;
  user: AuthUser | null;
  session: Session | null;
  mode: "demo" | "workspace";
  signInWithGoogle: () => Promise<{ error: string | null }>;
  signInWithEmail: (email: string) => Promise<{ error: string | null }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

function toUser(session: Session | null): AuthUser | null {
  if (!session?.user) return null;
  const meta = session.user.user_metadata ?? {};
  return {
    id: session.user.id,
    email: session.user.email ?? "",
    // Google returns full_name; fall back to the email local part so a name is always present.
    name: (meta.full_name as string) || (meta.name as string) || (session.user.email ?? "").split("@")[0] || "Reviewer",
    avatarUrl: (meta.avatar_url as string) ?? (meta.picture as string) ?? null,
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(Boolean(supabase));

  useEffect(() => {
    if (!supabase) return;
    let active = true;

    void supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      setSession(data.session);
      setLoading(false);
    });

    // Covers sign-in, sign-out, token refresh and the OAuth redirect landing back on the app.
    const { data: subscription } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      setLoading(false);
    });

    return () => {
      active = false;
      subscription.subscription.unsubscribe();
    };
  }, []);

  const value = useMemo<AuthState>(() => {
    const user = toUser(session);
    return {
      configured: Boolean(supabase),
      loading,
      user,
      session,
      mode: user ? "workspace" : "demo",
      async signInWithGoogle() {
        if (!supabase) return { error: "Authentication is not configured on this deployment." };
        const { error } = await supabase.auth.signInWithOAuth({
          provider: "google",
          options: { redirectTo: window.location.origin + window.location.pathname },
        });
        return { error: error?.message ?? null };
      },
      async signInWithEmail(email: string) {
        if (!supabase) return { error: "Authentication is not configured on this deployment." };
        const { error } = await supabase.auth.signInWithOtp({
          email,
          options: { emailRedirectTo: window.location.origin + window.location.pathname },
        });
        return { error: error?.message ?? null };
      },
      async signOut() {
        await supabase?.auth.signOut();
        setSession(null);
      },
    };
  }, [session, loading]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}

/** The bearer token the API validates. Null in demo mode, which the API treats as guest. */
export async function currentAccessToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}
