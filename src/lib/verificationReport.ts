import { Claim, VerificationResult } from "@/types/truthlens";

/**
 * Verification report export.
 *
 * "Export to PDF" was previously `window.print()` against the live app, which printed the
 * navbar, particle canvas and glass blur, and omitted evidence, reasoning, timings and model.
 * The report is now built as a self-contained document with its own print stylesheet, so what
 * reaches the printer is a compliance record rather than a screenshot of the UI.
 */

const escapeHtml = (value: unknown): string =>
  String(value ?? "").replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char] as string,
  );

/** Quote per RFC 4180 and neutralise leading =, +, -, @ so values cannot execute in a spreadsheet. */
const csvCell = (value: unknown): string => {
  const text = String(value ?? "");
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return `"${safe.replace(/"/g, '""')}"`;
};

const download = (content: string, mime: string, filename: string) => {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

export const reportRef = (result: VerificationResult): string => result.documentId ?? `unpersisted-${result.id}`;

/* ── JSON ─────────────────────────────────────────────────────────────────── */

export function exportJson(result: VerificationResult) {
  download(JSON.stringify(result, null, 2), "application/json", `truthlens_report_${reportRef(result)}.json`);
}

/* ── CSV ──────────────────────────────────────────────────────────────────── */

export function exportCsv(result: VerificationResult) {
  const headers = [
    "Claim ID", "Field", "Category", "Original AI Claim", "Verified Value", "Status", "Trust Score",
    "OCR", "Vision", "Layout", "Semantic", "Evidence Strength", "Signals Measured",
    "Pre-verification Risk", "Evidence Cited", "Evidence Retrieved", "Pages", "Reviewer", "Reviewer Notes", "Decision Reason",
  ];

  const rows = result.claims.map((claim) => [
    claim.id,
    claim.field,
    claim.category ?? "",
    claim.originalValue,
    claim.verifiedValue ?? "",
    claim.status,
    `${claim.trustScore}%`,
    signalCell(claim, "ocrAgreement"),
    signalCell(claim, "visionAgreement"),
    signalCell(claim, "layoutAgreement"),
    signalCell(claim, "semanticAgreement"),
    signalCell(claim, "evidenceStrength"),
    `${claim.confidenceBreakdown.measuredCount}/5`,
    claim.hallucinationRisk ? `${claim.hallucinationRisk.level} (${claim.hallucinationRisk.score}%)` : "",
    claim.retrieval?.citedCount ?? 0,
    claim.retrieval?.candidateCount ?? 0,
    [...new Set(claim.evidence.map((item) => item.pageNumber))].join(" "),
    claim.feedback ? "Human reviewer" : "Automated engine",
    claim.feedback?.reviewerNotes ?? "",
    claim.reason,
  ]);

  const csv = [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n");
  download(csv, "text/csv;charset=utf-8", `truthlens_report_${reportRef(result)}.csv`);
}

const signalCell = (claim: Claim, key: "ocrAgreement" | "visionAgreement" | "layoutAgreement" | "semanticAgreement" | "evidenceStrength") =>
  claim.confidenceBreakdown.unmeasured?.includes(key) ? "not measured" : `${claim.confidenceBreakdown[key]}%`;

/* ── Printable report ─────────────────────────────────────────────────────── */

/**
 * Opens a standalone, self-contained report document ready for Print → Save as PDF.
 * Returns false when the browser blocked the popup so the caller can tell the user.
 */
export function openPrintableReport(result: VerificationResult): boolean {
  const win = window.open("", "_blank", "width=1100,height=900");
  if (!win) return false;
  win.document.write(buildReportHtml(result));
  win.document.close();
  win.focus();
  return true;
}

function buildReportHtml(result: VerificationResult): string {
  const generated = new Date().toLocaleString();
  const created = new Date(result.createdAt).toLocaleString();
  const s = result.summary;

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TruthLens Verification Report — ${escapeHtml(result.fileName)}</title>
<style>
  :root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --ok:#047857; --warn:#b45309; --bad:#b91c1c; --info:#1d4ed8; }
  * { box-sizing: border-box; }
  body { font: 12px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: var(--ink); margin: 0; padding: 32px 40px; background: #fff; }
  h1 { font-size: 21px; margin: 0 0 2px; letter-spacing: -0.01em; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 26px 0 10px; padding-bottom: 5px; border-bottom: 1px solid var(--line); }
  .sub { color: var(--muted); font-size: 11px; }
  header { border-bottom: 2px solid var(--ink); padding-bottom: 12px; margin-bottom: 6px; display:flex; justify-content:space-between; align-items:flex-end; gap:24px; }
  .meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
  .meta div { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }
  .meta span { display:block; font-size: 9px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: 3px; }
  .meta b { font-size: 12px; font-weight: 600; word-break: break-word; }
  .kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
  .kpi { border: 1px solid var(--line); border-radius: 6px; padding: 10px; text-align: center; }
  .kpi b { display: block; font-size: 19px; line-height: 1.2; }
  .kpi span { font-size: 9px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 10.5px; }
  th { text-align: left; background: #f9fafb; border-bottom: 1px solid var(--line); padding: 7px 8px; font-size: 9px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  td { border-bottom: 1px solid var(--line); padding: 7px 8px; vertical-align: top; }
  .claim { border: 1px solid var(--line); border-radius: 7px; padding: 14px; margin-bottom: 12px; page-break-inside: avoid; }
  .claim-head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 8px; }
  .claim-head h3 { margin: 0; font-size: 13px; }
  .pill { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }
  .verified { color: var(--ok); } .corrected { color: var(--warn); } .unsupported { color: var(--bad); } .needs_review { color: var(--info); }
  .vals { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 9px; }
  .vals div { background: #f9fafb; border-radius: 5px; padding: 7px 9px; }
  .vals span { display:block; font-size: 9px; text-transform: uppercase; letter-spacing:.05em; color: var(--muted); margin-bottom: 2px; }
  .vals b { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; word-break: break-word; }
  .sig { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin: 9px 0; }
  .sig div { border: 1px solid var(--line); border-radius: 5px; padding: 6px; text-align: center; }
  .sig span { display:block; font-size: 8px; text-transform: uppercase; color: var(--muted); }
  .sig b { font-size: 12px; }
  .sig .na { color: var(--muted); font-weight: 500; font-size: 9px; }
  ul { margin: 5px 0 0; padding-left: 16px; }
  li { margin-bottom: 2px; }
  .ev { font-size: 10px; border-left: 2px solid var(--line); padding-left: 9px; margin-top: 5px; }
  .ev em { color: var(--muted); font-style: normal; }
  footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 9.5px; color: var(--muted); }
  .warn-banner { border: 1px solid var(--warn); color: var(--warn); border-radius: 6px; padding: 8px 11px; font-size: 10.5px; margin-top: 12px; }
  @media print { body { padding: 0; } .noprint { display: none; } h2 { page-break-after: avoid; } }
  .noprint { margin-bottom: 18px; }
  .noprint button { font: inherit; padding: 7px 15px; border-radius: 6px; border: 1px solid var(--ink); background: var(--ink); color: #fff; cursor: pointer; }
</style></head><body>

<div class="noprint"><button onclick="window.print()">Print / Save as PDF</button></div>

<header>
  <div>
    <h1>Verification Report</h1>
    <p class="sub">TruthLens AI · evidence-backed claim verification</p>
  </div>
  <div class="sub" style="text-align:right">
    <div>Report generated ${escapeHtml(generated)}</div>
    <div>Reference ${escapeHtml(reportRef(result))}</div>
  </div>
</header>

${result.persistence?.persisted === false ? `<div class="warn-banner"><b>Not stored.</b> ${escapeHtml(result.persistence.reason || "This verification was not written to the audit store, so it cannot be reconciled against server-side records.")}</div>` : ""}

<h2>Document</h2>
<div class="meta">
  <div><span>File</span><b>${escapeHtml(result.fileName)}</b></div>
  <div><span>Classified as</span><b>${escapeHtml(result.documentType)}</b></div>
  <div><span>Size</span><b>${escapeHtml(result.fileSizeKb)} KB</b></div>
  <div><span>Verified at</span><b>${escapeHtml(created)}</b></div>
  <div><span>Model</span><b>${escapeHtml(result.modelUsed)}</b></div>
  <div><span>Processing time</span><b>${escapeHtml(result.verificationTimeMs.toLocaleString())} ms</b></div>
  <div><span>Pages indexed</span><b>${escapeHtml(result.documentQuality?.pageCount ?? "—")}</b></div>
  <div><span>Mean legibility</span><b>${escapeHtml(result.documentQuality?.meanLegibility ?? "—")}%</b></div>
</div>

<h2>Verification summary</h2>
<div class="kpis">
  <div class="kpi"><b>${s.trustScore}%</b><span>Trust score</span></div>
  <div class="kpi"><b>${escapeHtml(s.riskLevel)}</b><span>Risk level</span></div>
  <div class="kpi"><b>${s.totalClaims}</b><span>Claims</span></div>
  <div class="kpi"><b class="verified">${s.verifiedCount}</b><span>Verified</span></div>
  <div class="kpi"><b class="corrected">${s.correctedCount}</b><span>Corrected</span></div>
  <div class="kpi"><b class="needs_review">${s.needsReviewCount}</b><span>Needs review</span></div>
</div>

<h2>Decision timeline (measured)</h2>
<table>
  <thead><tr><th style="width:22%">Stage</th><th style="width:10%">Duration</th><th style="width:16%">Timestamp</th><th>Detail</th></tr></thead>
  <tbody>${result.timeline
    .map(
      (event) => `<tr><td><b>${escapeHtml(event.title)}</b></td><td>${escapeHtml(event.durationMs)} ms</td><td>${escapeHtml(new Date(event.timestamp).toLocaleTimeString())}</td><td>${escapeHtml(event.detail)}</td></tr>`,
    )
    .join("")}</tbody>
</table>

<h2>Claims (${result.claims.length})</h2>
${result.claims.map(renderClaim).join("")}

<footer>
  Generated by TruthLens AI. Signals marked "not measured" were excluded from the trust score rather than
  substituted. Pre-verification risk is computed from document legibility and retrieval strength before the
  verifier runs, and is independent of the verification outcome.
  ${result.persistence?.persisted ? `Reconcilable against audit store record ${escapeHtml(result.documentId)}.` : "This report was produced from an unstored verification."}
</footer>

</body></html>`;
}

function renderClaim(claim: Claim): string {
  const b = claim.confidenceBreakdown;
  const unmeasured = new Set(b.unmeasured || []);
  const signal = (key: "ocrAgreement" | "visionAgreement" | "layoutAgreement" | "semanticAgreement" | "evidenceStrength", label: string) =>
    `<div><span>${label}</span>${unmeasured.has(key) ? '<b class="na">not measured</b>' : `<b>${b[key]}%</b>`}</div>`;

  const cited = claim.evidence.filter((item) => item.cited);

  return `<div class="claim">
  <div class="claim-head">
    <h3>${escapeHtml(claim.field)}${claim.category ? ` <span class="sub">· ${escapeHtml(claim.category)}</span>` : ""}</h3>
    <div>
      <span class="pill ${claim.status}">${escapeHtml(claim.status.replace("_", " "))}</span>
      <span class="pill">${claim.trustScore}% trust</span>
      ${claim.hallucinationRisk ? `<span class="pill">${escapeHtml(claim.hallucinationRisk.level)} pre-risk</span>` : ""}
    </div>
  </div>

  <div class="vals">
    <div><span>Original AI claim</span><b>${escapeHtml(claim.originalValue)}</b></div>
    <div><span>Verified value</span><b>${escapeHtml(claim.verifiedValue ?? "— withheld —")}</b></div>
  </div>

  <div class="sig">
    ${signal("ocrAgreement", "OCR")}${signal("visionAgreement", "Vision")}${signal("layoutAgreement", "Layout")}${signal("semanticAgreement", "Semantic")}${signal("evidenceStrength", "Evidence")}
  </div>

  <p style="margin:6px 0"><b>Decision reason.</b> ${escapeHtml(claim.reason)}</p>

  ${b.why?.length ? `<p style="margin:6px 0 0"><b>Why this score</b></p><ul>${b.why.map((why) => `<li>${escapeHtml(why)}</li>`).join("")}</ul>` : ""}

  ${claim.reasoning?.length ? `<p style="margin:8px 0 0"><b>Reasoning trace</b></p><ul>${claim.reasoning.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>` : ""}

  <p style="margin:8px 0 0"><b>Evidence</b> — ${cited.length} cited of ${claim.evidence.length} retrieved${claim.retrieval?.searched?.length ? `; searched ${escapeHtml(claim.retrieval.searched.join(", "))}` : ""}</p>
  ${
    cited.length === 0
      ? '<div class="ev"><em>No evidence was cited for this claim.</em></div>'
      : cited
          .map(
            (item) =>
              `<div class="ev"><em>${escapeHtml(item.source)} · retrieval score ${escapeHtml(item.confidence)}%${item.boundingBox ? ` · box (${item.boundingBox.x}, ${item.boundingBox.y}, ${item.boundingBox.width}×${item.boundingBox.height})` : " · no coordinates"}</em><br>${escapeHtml(item.text)}</div>`,
          )
          .join("")
  }

  ${
    claim.feedback
      ? `<p style="margin:8px 0 0"><b>Human decision.</b> ${escapeHtml(claim.feedback.status)}${claim.feedback.overrideValue ? ` → ${escapeHtml(claim.feedback.overrideValue)}` : ""} at ${escapeHtml(new Date(claim.feedback.timestamp).toLocaleString())}${claim.feedback.reviewerNotes ? `<br><em>${escapeHtml(claim.feedback.reviewerNotes)}</em>` : ""}</p>`
      : ""
  }
</div>`;
}
