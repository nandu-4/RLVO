import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  variant?: "default" | "light" | "strong";
  onClick?: () => void;
}

export default function GlassCard({
  children,
  className,
  hover = true,
  variant = "default",
  onClick,
}: GlassCardProps) {
  const variantClass =
    variant === "light" ? "glass-light" : variant === "strong" ? "glass-strong" : "glass";

  return (
    <div
      onClick={onClick}
      className={cn(
        variantClass,
        "rounded-2xl p-6",
        hover && "card-hover",
        onClick ? "cursor-pointer" : "cursor-default",
        className,
      )}
    >
      {children}
    </div>
  );
}
