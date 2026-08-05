import { describe, it, expect } from "vitest";
import { buildDocumentIndex } from "../api/_documentIndex.js";
import { retrieveEvidence } from "../api/_retrieval.js";
import { computeSignals, predictHallucinationRisk, buildClaimGraph } from "../api/_signals.js";
import { assembleClaim, parseUpstreamClaims } from "../api/_truthlens.js";

const index = buildDocumentIndex({
  documentType: "invoice",
  pageCount: 1,
  blocks: [
    { page: 1, text: "ORACLE CORPORATION", region: "header", legibility: 99, box_2d: [18, 115, 42, 610] },
    { page: 1, text: "Total: $13,511.00", region: "table", legibility: 98, box_2d: [374, 115, 392, 320] },
    { page: 1, text: "Payment Terms: Net 30", region: "footer", legibility: 45, box_2d: [430, 115, 446, 350] },
  ],
});

const assemble = (field: string, value: string, verdict: Parameters<typeof assembleClaim>[0]["verdict"]) => {
  const retrieval = retrieveEvidence(index, field, value);
  return assembleClaim({
    claim: { field, value }, index: 0, retrieval, verdict, quality: index.quality,
    risk: predictHallucinationRisk(field, value, retrieval.candidates, index.quality),
  });
};

describe("claim intake", () => {
  it("keeps well-formed claims and caps the batch", () => {
    expect(parseUpstreamClaims([{ field: "Vendor", value: "Oracle" }])).toHaveLength(1);
    expect(parseUpstreamClaims(Array.from({ length: 200 }, (_, i) => ({ field: `f${i}`, value: "v" })))).toHaveLength(60);
  });

  it("drops junk, blanks and non-arrays without throwing", () => {
    expect(parseUpstreamClaims([{ field: "", value: "x" }, { field: "a", value: "" }, null, 42])).toHaveLength(0);
    expect(parseUpstreamClaims([{ field: "Producer", value: "pdfTeX-1.40.25" }])).toHaveLength(0);
    expect(parseUpstreamClaims("nope" as unknown)).toHaveLength(0);
  });
});

describe("decision guardrails", () => {
  it("verifies a true claim that the provider grounded in cited evidence", () => {
    const c = assemble("Total", "$13,511.00", {
      field: "Total", status: "verified", reason: "matches", evidenceIds: ["block-2"], visionAgreement: 100,
    });
    expect(c.status).toBe("verified");
    expect(c.verifiedValue).toBe("$13,511.00");
    expect(c.trustScore).toBeGreaterThan(60);
  });

  it("downgrades verified to needs_review when no cited evidence resolves", () => {
    const c = assemble("Total", "$13,511.00", {
      field: "Total", status: "verified", reason: "trust me", evidenceIds: [],
    });
    expect(c.status).toBe("needs_review");
  });

  it("discards a citation the retrieval engine never returned", () => {
    const c = assemble("Total", "$13,511.00", {
      field: "Total", status: "verified", reason: "x", evidenceIds: ["block-9999"],
    });
    expect(c.status).toBe("needs_review");
    expect(c.evidence.every((e) => !e.cited)).toBe(true);
  });

  it("drops a correction that appears in no cited block", () => {
    const c = assemble("Vendor", "Microsoft Corporation", {
      field: "Vendor", status: "corrected", verified: "TOTALLY MADE UP LTD", reason: "x", evidenceIds: ["block-1"],
    });
    expect(c.verifiedValue).toBeUndefined();
    expect(c.reason).toMatch(/could not be grounded/i);
  });

  it("accepts a correction that is verbatim in cited evidence", () => {
    const c = assemble("Vendor", "Microsoft Corporation", {
      field: "Vendor", status: "corrected", verified: "ORACLE CORPORATION", reason: "header disagrees", evidenceIds: ["block-1"],
    });
    expect(c.status).toBe("corrected");
    expect(c.verifiedValue).toBe("ORACLE CORPORATION");
  });

  it("returns needs_review when the provider skipped the claim entirely", () => {
    const c = assemble("Vendor", "Microsoft Corporation", undefined);
    expect(c.status).toBe("needs_review");
  });

  it("rejects an unknown status rather than passing it through", () => {
    const c = assemble("Total", "$13,511.00", {
      field: "Total", status: "definitely-true" as never, reason: "x", evidenceIds: ["block-2"],
    });
    expect(["verified", "corrected", "unsupported", "needs_review"]).toContain(c.status);
    expect(c.status).toBe("needs_review");
  });
});

describe("trust signals", () => {
  const signalsFor = (visionAgreement?: number) => {
    const retrieval = retrieveEvidence(index, "Total", "$13,511.00");
    return computeSignals({
      field: "Total", claimedValue: "$13,511.00",
      cited: retrieval.candidates.slice(0, 1), retrieved: retrieval.candidates,
      providerVisionAgreement: visionAgreement, quality: index.quality,
    });
  };

  it("marks vision as unmeasured when the provider omits it, instead of substituting", () => {
    const s = signalsFor(undefined);
    expect(s.visionAgreement.measured).toBe(false);
    expect(s.measuredCount).toBe(4);
    // The unmeasured signal must not silently become another signal's value.
    expect(s.visionAgreement.value).toBe(0);
    expect(s.finalTrustScore).toBeGreaterThan(0);
  });

  it("excludes unmeasured signals from the mean rather than scoring them zero", () => {
    const withoutVision = signalsFor(undefined).finalTrustScore;
    const withPerfectVision = signalsFor(100).finalTrustScore;
    // If unmeasured counted as 0, adding a perfect vision score would move the mean enormously.
    expect(withPerfectVision).toBeGreaterThanOrEqual(withoutVision);
    expect(withoutVision).toBeGreaterThan(40);
  });

  it("gives every signal a stated basis and a why narrative", () => {
    const s = signalsFor(90);
    for (const key of ["ocrAgreement", "visionAgreement", "layoutAgreement", "semanticAgreement", "evidenceStrength"] as const) {
      expect(s[key].basis.length).toBeGreaterThan(10);
    }
    expect(s.why.length).toBeGreaterThan(0);
  });

  it("scores zero with no evidence at all", () => {
    const s = computeSignals({ field: "X", claimedValue: "Y", cited: [], retrieved: [], quality: index.quality });
    expect(s.finalTrustScore).toBe(0);
    expect(s.why.join(" ")).toMatch(/no evidence/i);
  });
});

describe("pre-verification hallucination risk", () => {
  it("is HIGH when nothing was retrieved", () => {
    const r = predictHallucinationRisk("Shipping Weight", "42 kg", [], index.quality);
    expect(r.level).toBe("HIGH");
    expect(r.reasons.join(" ")).toMatch(/no candidate evidence/i);
  });

  it("is LOW for a strong exact match", () => {
    const retrieval = retrieveEvidence(index, "Total", "$13,511.00");
    expect(predictHallucinationRisk("Total", "$13,511.00", retrieval.candidates, index.quality).level).toBe("LOW");
  });

  it("raises risk when the matching region is barely legible", () => {
    const retrieval = retrieveEvidence(index, "Payment Terms", "Net 30");
    const r = predictHallucinationRisk("Payment Terms", "Net 30", retrieval.candidates, index.quality);
    expect(r.reasons.join(" ")).toMatch(/legibility/i);
  });

  it("never depends on the verification outcome — same inputs, same result", () => {
    const retrieval = retrieveEvidence(index, "Total", "$13,511.00");
    const a = predictHallucinationRisk("Total", "$13,511.00", retrieval.candidates, index.quality);
    const b = predictHallucinationRisk("Total", "$13,511.00", retrieval.candidates, index.quality);
    expect(a).toEqual(b);
  });
});

describe("claim relation graph", () => {
  it("links claims proven by the same block most strongly", () => {
    const block = index.blocks[0];
    const rels = buildClaimGraph([
      { id: "c1", field: "Vendor", value: "Oracle", blocks: [block] },
      { id: "c2", field: "Vendor Address", value: "Redwood", blocks: [block] },
    ]);
    expect(rels[0].kind).toBe("shared-evidence");
    expect(rels[0].strength).toBe(1);
  });

  it("never links a claim to itself", () => {
    const rels = buildClaimGraph([{ id: "c1", field: "A", value: "1", blocks: [index.blocks[0]] }]);
    expect(rels).toHaveLength(0);
  });
});
