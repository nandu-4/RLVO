import { useState } from "react";
import { Link } from "react-router-dom";
import { LogOut, User, Loader2, ShieldCheck, FlaskConical, LogIn } from "lucide-react";
import { useAuth } from "@/integrations/auth";

export default function AccountMenu() {
  const { configured, loading, user, signOut } = useAuth();
  const [open, setOpen] = useState(false);

  if (loading) return <div className="glass-light rounded-xl px-3 py-2 flex items-center gap-2 text-xs text-muted-foreground"><Loader2 className="w-3.5 h-3.5 animate-spin" /> Checking session…</div>;
  if (!configured) return <span className="glass-light rounded-xl px-3 py-2 text-[11px] text-muted-foreground flex items-center gap-1.5"><FlaskConical className="w-3.5 h-3.5 text-warning" /> Demo mode</span>;

  if (!user) {
    // One clean entry point: the dedicated /login page owns the Google + email options, so the
    // navbar stays uncluttered and mobile does not inherit a two-control form.
    return (
      <Link to="/login" className="btn-primary text-xs py-2 px-4 flex items-center gap-1.5">
        <LogIn className="w-3.5 h-3.5" /> Sign in
      </Link>
    );
  }

  return (
    <div className="relative">
      <button onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-haspopup="menu" className="glass-light rounded-xl pl-1.5 pr-3 py-1.5 flex items-center gap-2 border border-border hover:border-primary/50 transition-colors">
        {user.avatarUrl ? <img src={user.avatarUrl} alt="" className="w-7 h-7 rounded-lg object-cover" /> : <span className="w-7 h-7 rounded-lg bg-primary/15 text-primary flex items-center justify-center"><User className="w-4 h-4" /></span>}
        <span className="text-left hidden sm:block"><span className="block text-xs font-semibold text-foreground leading-tight">{user.name}</span><span className="block text-[10px] text-success leading-tight flex items-center gap-1"><ShieldCheck className="w-3 h-3" /> Workspace mode</span></span>
      </button>
      {open && <><div className="fixed inset-0 z-40" onClick={() => setOpen(false)} aria-hidden="true" /><div role="menu" className="absolute right-0 mt-2 w-64 glass-strong rounded-xl border border-border/60 shadow-2xl z-50 p-3"><div className="pb-3 mb-2 border-b border-border/40"><p className="text-sm font-bold text-foreground truncate">{user.name}</p><p className="text-[11px] text-muted-foreground truncate">{user.email}</p></div><button onClick={async () => { setOpen(false); await signOut(); }} className="btn-secondary w-full text-xs py-2 flex items-center justify-center gap-1.5"><LogOut className="w-3 h-3" /> Sign out</button></div></>}
    </div>
  );
}

export function SignInRequired({ feature }: { feature: string }) {
  const { configured } = useAuth();
  return (
    <div className="glass rounded-2xl border border-border/60 p-10 text-center max-w-2xl mx-auto">
      <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center mx-auto mb-5 text-primary"><User className="w-7 h-7" /></div>
      <h2 className="text-lg font-bold">{feature} needs an account</h2>
      <p className="text-sm text-muted-foreground mt-2 leading-relaxed">Sign in to save your work and access this area.</p>
      {!configured && <p className="text-xs text-warning mt-6">Authentication is not configured on this deployment.</p>}
      {configured && (
        <Link to="/login" className="btn-primary text-xs py-2.5 px-5 inline-block mt-6">Sign in</Link>
      )}
    </div>
  );
}
