import { useMemo } from "react";
import { Share2 } from "lucide-react";
import { Claim, ClaimRelation } from "@/types/truthlens";
import GlassCard from "./GlassCard";

interface ClaimRelationGraphProps {
  claims: Claim[];
  relations: ClaimRelation[];
  selectedClaimId?: string;
  onSelectClaim: (claim: Claim) => void;
}

const STATUS_FILL: Record<string, string> = {
  verified: "hsl(var(--success))",
  corrected: "hsl(var(--warning))",
  unsupported: "hsl(var(--danger))",
  needs_review: "hsl(var(--accent))",
};

const EDGE_STYLE: Record<ClaimRelation["kind"], { stroke: string; dash?: string; label: string }> = {
  "shared-evidence": { stroke: "hsl(var(--success))", label: "same evidence block" },
  "same-region": { stroke: "hsl(var(--primary))", dash: "4 3", label: "same page region" },
  "same-page": { stroke: "hsl(var(--muted-foreground))", dash: "2 4", label: "same page" },
  lexical: { stroke: "hsl(var(--accent))", dash: "1 4", label: "related field name" },
};

/**
 * Lays claims out on a circle and draws the relationships the server derived from where each
 * claim's evidence physically sits in the document. Nothing here knows what an invoice or a
 * resume is — the edges come from shared blocks, shared regions, and shared pages.
 */
export default function ClaimRelationGraph({ claims, relations, selectedClaimId, onSelectClaim }: ClaimRelationGraphProps) {
  const size = 340;
  const radius = size / 2 - 54;
  const centre = size / 2;

  const nodes = useMemo(
    () =>
      claims.map((claim, index) => {
        const angle = (index / Math.max(claims.length, 1)) * Math.PI * 2 - Math.PI / 2;
        return { claim, x: centre + radius * Math.cos(angle), y: centre + radius * Math.sin(angle) };
      }),
    [claims, centre, radius],
  );

  const positions = useMemo(() => new Map(nodes.map((node) => [node.claim.id, node])), [nodes]);
  const kindsPresent = useMemo(() => [...new Set(relations.map((relation) => relation.kind))], [relations]);

  if (claims.length < 2) return null;

  return (
    <GlassCard hover={false} className="p-5">
      <div className="flex items-center gap-2 mb-1">
        <Share2 className="w-4 h-4 text-primary" />
        <h3 className="text-sm font-semibold">Claim relation graph</h3>
      </div>
      <p className="text-[11px] text-muted-foreground mb-3">
        {relations.length === 0
          ? "No relationships were found — each claim was proven by a different part of the document."
          : `${relations.length} relationship(s) derived from where each claim's evidence sits on the page.`}
      </p>

      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[340px] mx-auto" role="img" aria-label="Claim relation graph">
          {relations.map((relation, index) => {
            const from = positions.get(relation.from);
            const to = positions.get(relation.to);
            if (!from || !to) return null;
            const style = EDGE_STYLE[relation.kind];
            const highlighted = selectedClaimId === relation.from || selectedClaimId === relation.to;
            return (
              <line
                key={`${relation.from}-${relation.to}-${index}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={style.stroke}
                strokeWidth={highlighted ? 2.2 : 1 + relation.strength}
                strokeDasharray={style.dash}
                opacity={selectedClaimId ? (highlighted ? 0.95 : 0.18) : 0.5}
              />
            );
          })}

          {nodes.map((node) => {
            const isSelected = node.claim.id === selectedClaimId;
            return (
              <g
                key={node.claim.id}
                onClick={() => onSelectClaim(node.claim)}
                className="cursor-pointer"
                role="button"
                aria-label={`${node.claim.field}: ${node.claim.status.replace("_", " ")}`}
              >
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={isSelected ? 11 : 8}
                  fill={STATUS_FILL[node.claim.status] || STATUS_FILL.needs_review}
                  stroke={isSelected ? "hsl(var(--foreground))" : "transparent"}
                  strokeWidth={2}
                  opacity={selectedClaimId && !isSelected ? 0.5 : 1}
                />
                <text
                  x={node.x}
                  y={node.y + (node.y < centre ? -16 : 22)}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 9, fontWeight: isSelected ? 700 : 500 }}
                >
                  {node.claim.field.length > 18 ? `${node.claim.field.slice(0, 17)}…` : node.claim.field}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {kindsPresent.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-3 text-[10px] text-muted-foreground">
          {kindsPresent.map((kind) => (
            <span key={kind} className="flex items-center gap-1.5">
              <span className="w-4 h-px" style={{ backgroundColor: EDGE_STYLE[kind].stroke }} />
              {EDGE_STYLE[kind].label}
            </span>
          ))}
        </div>
      )}
    </GlassCard>
  );
}
