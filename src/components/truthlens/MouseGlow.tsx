import { useEffect, useRef } from "react";
import { useReducedMotion } from "@/hooks/useReducedMotion";

export default function MouseGlow() {
  const glowRef = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (reducedMotion) return;
    const handleMove = (e: MouseEvent) => {
      if (glowRef.current) {
        glowRef.current.style.left = `${e.clientX}px`;
        glowRef.current.style.top = `${e.clientY}px`;
      }
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, [reducedMotion]);

  return (
    <div
      ref={glowRef}
      className="fixed pointer-events-none z-0 -translate-x-1/2 -translate-y-1/2"
      style={{
        width: "600px",
        height: "600px",
        background:
          "radial-gradient(circle, hsla(220, 90%, 56%, 0.08) 0%, hsla(270, 76%, 53%, 0.04) 40%, transparent 70%)",
        filter: "blur(1px)",
      }}
      aria-hidden="true"
    />
  );
}
