import { describe, it, expect } from "vitest";
import { buildDocumentIndex, tokenize } from "../api/_documentIndex.js";
import { retrieveEvidence, diceSimilarity, tokenOverlap } from "../api/_retrieval.js";

/** Mirrors the real invoice used in live end-to-end testing. */
const invoice = () =>
  buildDocumentIndex({
    documentType: "invoice",
    pageCount: 1,
    blocks: [
      { page: 1, text: "ORACLE CORPORATION", region: "header", legibility: 99, box_2d: [18, 115, 42, 610] },
      { page: 1, text: "500 Oracle Parkway, Redwood City, CA 94065", region: "header", legibility: 98, box_2d: [50, 115, 68, 540] },
      { page: 1, text: "Invoice Number: INV-2024-8891", region: "body", legibility: 97, box_2d: [124, 115, 140, 395] },
      { page: 1, text: "Invoice Date: 14 March 2024", region: "body", legibility: 97, box_2d: [145, 115, 161, 370] },
      { page: 1, text: "Subtotal: $11,450.00", region: "body", legibility: 96, box_2d: [330, 115, 346, 330] },
      { page: 1, text: "GST (18%): $2,061.00", region: "body", legibility: 96, box_2d: [352, 115, 368, 330] },
      { page: 1, text: "Total: $13,511.00", region: "table", legibility: 98, box_2d: [374, 115, 392, 320] },
      { page: 1, text: "Payment Terms: Net 30", region: "footer", legibility: 97, box_2d: [430, 115, 446, 350] },
    ],
  });

describe("document index", () => {
  it("indexes blocks with normalised coordinates and quality metrics", () => {
    const index = invoice();
    expect(index.documentType).toBe("invoice");
    expect(index.blocks).toHaveLength(8);
    expect(index.quality.boundingBoxCoverage).toBe(1);
    expect(index.quality.meanLegibility).toBeGreaterThan(90);
    for (const block of index.blocks) {
      expect(block.boundingBox!.y + block.boundingBox!.height).toBeLessThanOrEqual(100.5);
    }
  });

  it("rejects PDF container junk — the spec's hard rule", () => {
    const index = buildDocumentIndex({
      documentType: "invoice",
      pageCount: 1,
      blocks: [
        { page: 1, text: "Total: $99.00", region: "body", legibility: 95 },
        { page: 1, text: "%PDF-1.4", region: "body", legibility: 90 },
        { page: 1, text: "12 0 obj", region: "body", legibility: 90 },
        { page: 1, text: "/Producer (pdfTeX-1.40.25)", region: "body", legibility: 90 },
        { page: 1, text: "endstream endobj", region: "body", legibility: 90 },
        { page: 1, text: "startxref", region: "body", legibility: 90 },
      ],
    });
    expect(index.blocks).toHaveLength(1);
    expect(index.blocks[0].text).toBe("Total: $99.00");
  });

  it("treats a missing legibility as moderate, never as perfect", () => {
    const index = buildDocumentIndex({ documentType: "x", pageCount: 1, blocks: [{ page: 1, text: "Hello world", region: "body" }] });
    expect(index.blocks[0].legibility).toBe(60);
  });

  it("keeps the most legible copy of a duplicated block", () => {
    const index = buildDocumentIndex({
      documentType: "x", pageCount: 1,
      blocks: [
        { page: 1, text: "Total: $10", region: "body", legibility: 40 },
        { page: 1, text: "Total: $10", region: "body", legibility: 95 },
      ],
    });
    expect(index.blocks).toHaveLength(1);
    expect(index.blocks[0].legibility).toBe(95);
  });
});

describe("evidence retrieval engine", () => {
  it("ranks the exact-matching block first", () => {
    const report = retrieveEvidence(invoice(), "Total", "$13,511.00");
    expect(report.candidates[0].block.text).toBe("Total: $13,511.00");
    expect(report.candidates[0].valueSimilarity).toBeGreaterThan(0.7);
  });

  it("retrieves the vendor from the header for a wrong claim, so it can be corrected", () => {
    const report = retrieveEvidence(invoice(), "Vendor", "Microsoft Corporation");
    expect(report.candidates.length).toBeGreaterThan(0);
    expect(report.candidates.map((c) => c.block.text)).toContain("ORACLE CORPORATION");
  });

  it("returns nothing for a fact absent from the document", () => {
    const report = retrieveEvidence(invoice(), "Shipping Weight", "42 kg");
    expect(report.candidates).toHaveLength(0);
  });

  it("distinguishes similar money values rather than collapsing them", () => {
    const report = retrieveEvidence(invoice(), "Subtotal", "$11,450.00");
    expect(report.candidates[0].block.text).toBe("Subtotal: $11,450.00");
  });

  it("reports which surfaces were searched and which strategies hit", () => {
    const report = retrieveEvidence(invoice(), "Payment Terms", "Net 30");
    expect(report.searched).toContain("Transcribed text");
    expect(report.strategiesHit.length).toBeGreaterThan(0);
    expect(report.candidates[0].strategies).toContain("value-match");
  });

  it("fires the numeric strategy when a claim carries figures", () => {
    const report = retrieveEvidence(invoice(), "GST", "2,061.00");
    expect(report.strategiesHit).toContain("numeric");
  });

  it("survives an empty index without throwing", () => {
    const empty = buildDocumentIndex({ documentType: "x", pageCount: 1, blocks: [] });
    expect(retrieveEvidence(empty, "Vendor", "Acme").candidates).toHaveLength(0);
  });
});

describe("similarity primitives", () => {
  it("scores identical strings 1 and unrelated strings near 0", () => {
    expect(diceSimilarity("ORACLE", "ORACLE")).toBe(1);
    expect(diceSimilarity("ORACLE", "zzzzzz")).toBeLessThan(0.2);
  });

  it("is tolerant of OCR-style noise", () => {
    expect(diceSimilarity("ORACLE CORPORATION", "0RACLE CORPORATION")).toBeGreaterThan(0.85);
  });

  it("computes token overlap symmetrically", () => {
    const a = tokenize("Invoice Number INV-2024-8891");
    const b = tokenize("Invoice Number: INV-2024-8891");
    expect(tokenOverlap(a, b)).toBeGreaterThan(0.5);
    expect(tokenOverlap(a, b)).toBe(tokenOverlap(b, a));
  });

  it("strips stopwords so they cannot inflate a match", () => {
    expect(tokenize("the total of the invoice")).not.toContain("the");
  });
});
