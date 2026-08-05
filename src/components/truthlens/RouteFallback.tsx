import { Shield } from "lucide-react";

/**
 * Shown while a lazily-loaded route downloads.
 *
 * Deliberately holds the page's real chrome — background, centred shield, full viewport height —
 * so a route change reads as continuous rather than a flash of empty page followed by a jump.
 * No spinner: on a fast connection the chunk arrives in well under the ~200ms at which a spinner
 * starts to register, and a spinner that flickers in and straight out feels worse than a calm hold.
 */
export default function RouteFallback() {
  return (
    <div className="min-h-screen aurora-bg flex items-center justify-center" role="status" aria-live="polite">
      <div className="flex flex-col items-center gap-3">
        <Shield className="w-9 h-9 text-primary animate-pulse-glow" aria-hidden="true" />
        <span className="sr-only">Loading page</span>
      </div>
    </div>
  );
}
