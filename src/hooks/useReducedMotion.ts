import { useEffect, useState } from "react";

/**
 * Tracks the OS "reduce motion" preference, and reacts if the user changes it live.
 *
 * CSS handles declarative animation, but canvas render loops and pointer-driven effects run in
 * JavaScript and keep going regardless of the media query — so the components that own them have
 * to opt out explicitly. For users with vestibular sensitivity a drifting particle field is a
 * genuine barrier, not a nicety.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false,
  );

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
}
