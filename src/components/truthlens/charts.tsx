import { useState } from "react";

/**
 * Chart primitives for the TruthLens dashboards.
 *
 * DESIGN NOTE — why labelled rows rather than stacked bars or a pie:
 * the app's status palette was validated for colour-vision separation and warning↔success sits at
 * ΔE 7.5 (protan), inside the 6–8 floor band. That band is only usable with secondary encoding.
 * The theme is fixed, so instead of repainting it these components never place status colours
 * adjacent: every value carries its own text label and number, colour is reinforcement only, and
 * segments are separated by a surface gap. Identity is therefore never colour-alone.
 */

const fmt = (value: number) => value.toLocaleString();

/* ── Labelled magnitude rows ─────────────────────────────────────────────── */

export interface BarRow {
  label: string;
  value: number;
  /** Tailwind background class. Reinforcement only — the label carries identity. */
  tone?: string;
  hint?: string;
  suffix?: string;
}

export function BarRows({ rows, emptyLabel = "No data yet" }: { rows: BarRow[]; emptyLabel?: string }) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground py-4 text-center">{emptyLabel}</p>;
  const peak = Math.max(...rows.map((row) => row.value), 1);

  return (
    <div className="space-y-2.5">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="flex items-baseline justify-between gap-3 mb-1">
            <span className="text-xs text-foreground truncate" title={row.hint || row.label}>
              {row.label}
            </span>
            <span className="text-xs font-bold text-foreground tabular-nums shrink-0">
              {fmt(row.value)}
              {row.suffix}
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-dark overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${row.tone || "bg-primary"}`}
              style={{ width: `${Math.max(2, (row.value / peak) * 100)}%` }}
            />
          </div>
          {row.hint && <p className="text-[10px] text-muted-foreground mt-1 leading-relaxed">{row.hint}</p>}
        </div>
      ))}
    </div>
  );
}

/* ── Single-series trend ─────────────────────────────────────────────────── */

export interface TrendPoint {
  date: string;
  documents: number;
  averageTrustScore: number | null;
  needsReview: number;
}

/**
 * One measure, one axis, one hue — so no legend and no colour-identity problem.
 * Days with no runs are plotted as gaps rather than zero, because "we ran nothing" and
 * "we scored zero" are different facts and a zero would lie about the average.
 */
export function TrendChart({ points, height = 132 }: { points: TrendPoint[]; height?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const withRuns = points.filter((point) => point.documents > 0);

  if (withRuns.length === 0) {
    return <p className="text-xs text-muted-foreground py-8 text-center">No verifications in this window yet.</p>;
  }

  const width = 640;
  const padY = 14;
  const x = (index: number) => (index / Math.max(points.length - 1, 1)) * width;
  const y = (score: number) => padY + (1 - score / 100) * (height - padY * 2);

  // Break the path wherever a day has no runs, instead of interpolating through absence.
  const segments: Array<Array<{ x: number; y: number }>> = [];
  let current: Array<{ x: number; y: number }> = [];
  points.forEach((point, index) => {
    if (point.averageTrustScore === null) {
      if (current.length > 0) segments.push(current);
      current = [];
      return;
    }
    current.push({ x: x(index), y: y(point.averageTrustScore) });
  });
  if (current.length > 0) segments.push(current);

  const active = hover !== null ? points[hover] : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height }}
        role="img"
        aria-label={`Average trust score across ${points.length} days`}
        onMouseLeave={() => setHover(null)}
      >
        {[0, 50, 100].map((tick) => (
          <g key={tick}>
            <line x1={0} x2={width} y1={y(tick)} y2={y(tick)} stroke="hsl(var(--border))" strokeWidth={1} opacity={0.5} />
            <text x={2} y={y(tick) - 3} className="fill-muted-foreground" style={{ fontSize: 9 }}>
              {tick}%
            </text>
          </g>
        ))}

        {segments.map((segment, index) => (
          <polyline
            key={index}
            points={segment.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}

        {segments.flat().map((point, index) => (
          <circle key={index} cx={point.x} cy={point.y} r={3} fill="hsl(var(--primary))" stroke="hsl(var(--card))" strokeWidth={1.5} />
        ))}

        {active?.averageTrustScore !== null && active !== null && (
          <line x1={x(hover as number)} x2={x(hover as number)} y1={padY} y2={height - padY} stroke="hsl(var(--accent))" strokeWidth={1} strokeDasharray="3 3" />
        )}

        {/* Hit targets are wider than the marks so hovering is comfortable. */}
        {points.map((point, index) => (
          <rect
            key={point.date}
            x={x(index) - width / points.length / 2}
            y={0}
            width={width / points.length}
            height={height}
            fill="transparent"
            onMouseEnter={() => setHover(index)}
          />
        ))}
      </svg>

      <div className="flex items-center justify-between text-[10px] text-muted-foreground mt-1">
        <span>{points[0]?.date}</span>
        <span className="text-primary font-semibold">Average trust score per day</span>
        <span>{points[points.length - 1]?.date}</span>
      </div>

      {active && (
        <div className="mt-2 glass-light rounded-lg px-3 py-2 text-[11px] flex flex-wrap gap-x-4 gap-y-1">
          <span className="font-mono text-muted-foreground">{active.date}</span>
          {active.documents === 0 ? (
            <span className="text-muted-foreground">No verifications</span>
          ) : (
            <>
              <span className="text-foreground">
                <b>{active.averageTrustScore}%</b> avg trust
              </span>
              <span className="text-muted-foreground">{fmt(active.documents)} document(s)</span>
              {active.needsReview > 0 && <span className="text-accent">{fmt(active.needsReview)} needing review</span>}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Stat tile ───────────────────────────────────────────────────────────── */

export function StatTile({
  label,
  value,
  hint,
  tone = "text-foreground",
  icon,
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="glass-light rounded-xl p-4">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">{label}</span>
        {icon && <span className="text-muted-foreground/70">{icon}</span>}
      </div>
      <div className={`text-2xl font-black tabular-nums ${tone}`}>{value}</div>
      {hint && <p className="text-[10px] text-muted-foreground mt-1 leading-relaxed">{hint}</p>}
    </div>
  );
}
