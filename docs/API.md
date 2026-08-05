# TruthLens AI — API Reference

All endpoints are `POST` (except `/api/health`), accept and return JSON, and are same-origin.

## Authentication

There is none. Tenancy is carried by an anonymous workspace token:

```
x-truthlens-workspace: <43-character URL-safe token>
```

The browser mints this on first use (`src/lib/workspace.ts`). The server stores only its SHA-256
hash. Omit it and endpoints still work where they can — verification runs, but nothing is stored
and every response says so in `persistence.reason`.

> The token is a bearer secret. Whoever holds it has the workspace, and it cannot be recovered.

## Errors

Every failure returns `{ "error": "<human-readable message>" }`. Messages that would expose
infrastructure (database URLs, keys, stack frames) are replaced with a generic message plus a
reference id; the full detail is logged server-side only.

| Status | Meaning |
|---|---|
| 400 | Malformed request — the message names the offending field |
| 401 | Workspace required for this operation |
| 403 | Resource belongs to another workspace |
| 404 | Resource not found |
| 413 | Document exceeds the 4.5 MB request limit |
| 422 | Verification could not complete (includes provider quota exhaustion) |
| 429 | Rate limited; `Retry-After` header set |
| 501 | Named provider has no configured adapter |
| 503 | Persistence not configured on this deployment |

---

## `POST /api/verify-document`

The core endpoint. Verifies supplied claims against a document.

```jsonc
{
  "image": "data:application/pdf;base64,…",   // or a raw base64 string
  "fileName": "invoice.pdf",
  "modelUsed": "gemini-2.5",
  "upstreamClaims": [
    { "field": "Vendor", "value": "Microsoft Corporation" },
    { "field": "Total",  "value": "$13,511.00" }
  ],
  "jobId": "uuid"                              // optional, links to a batch job
}
```

**Response** (abridged):

```jsonc
{
  "documentId": "uuid | null",                 // null when not persisted
  "documentType": "invoice",                   // classified, never assumed
  "modelUsed": "gemini-2.5-flash",             // what actually ran
  "summary": { "totalClaims": 4, "verifiedCount": 2, "correctedCount": 1,
               "unsupportedCount": 1, "needsReviewCount": 0,
               "trustScore": 64, "riskLevel": "MEDIUM" },
  "claims": [{
    "id": "uuid", "field": "Vendor", "originalValue": "Microsoft Corporation",
    "verifiedValue": "ORACLE CORPORATION", "status": "corrected", "trustScore": 59,
    "reason": "Header on page 1 disagrees with the claim.",
    "confidenceBreakdown": {
      "ocrAgreement": 88, "visionAgreement": 100, "layoutAgreement": 72,
      "semanticAgreement": 64, "evidenceStrength": 55, "finalTrustScore": 59,
      "measuredCount": 4,                      // of 5
      "unmeasured": [],                        // signals excluded from the mean
      "basis": { "ocrAgreement": "Best text similarity 92% …" },
      "why": ["1 evidence item cited from page 1.", "…"]
    },
    "hallucinationRisk": { "level": "LOW", "score": 12, "reasons": ["…"] },
    "retrieval": { "searched": ["Transcribed text","Headers"], "strategies": ["value-match"],
                   "candidateCount": 2, "citedCount": 1 },
    "evidence": [{ "id": "block-1", "text": "ORACLE CORPORATION", "pageNumber": 1,
                   "boundingBox": { "x": 11.5, "y": 1.8, "width": 49.5, "height": 2.4 },
                   "retrievedBy": ["value-match"], "cited": true, "confidence": 40 }],
    "reasoning": ["Planning: …", "Evidence Search: …", "Reflection: …", "Decision: …"]
  }],
  "relations": [{ "from": "…", "to": "…", "kind": "shared-evidence", "strength": 1 }],
  "documentQuality": { "blockCount": 23, "pageCount": 1, "meanLegibility": 97 },
  "timeline": [{ "step": "evidence_retrieval", "durationMs": 8, "detail": "…" }],
  "persistence": { "persisted": true, "mode": "workspace", "reason": null }
}
```

`status` is always one of `verified` · `corrected` · `unsupported` · `needs_review`.
Bounding boxes are always percentages of the page, top-left origin.

Rate limit: `VERIFY_RATE_LIMIT` (default 10) per minute. Max duration 60s.

---

## `POST /api/extract-claims`

Self-check mode: proposes the document's business facts as atomic claims. Returns a `caveat`
field — the proposer and verifier share a failure mode, so this is weaker than checking another
system's output.

```jsonc
// → { "claims": [{ "field": "Invoice Number", "value": "INV-2024-8891", "category": "Financial" }],
//     "extractionTimeMs": 4200, "caveat": "…" }
```

## `POST /api/review-claim`

Records a human decision. Requires a workspace; validates that the claim belongs to it before any
write. `reviewerName` is self-declared — the audit trail records who *said* they decided.

```jsonc
{ "documentId": "uuid", "claimId": "uuid",
  "feedback": { "status": "overridden", "overrideValue": "12.5 kg",
                "reviewerNotes": "found on packing slip", "reviewerName": "Nandini" } }
```

`status` is `approved` · `rejected` · `overridden`. An override must supply `overrideValue`;
it updates the claim, resolves the review task, and appends to `audit_trails`.

## `POST /api/review-queue`

| action | Body | Returns |
|---|---|---|
| `list` | `{ status? }` | Open work items with triage context |
| `assign` | `{ taskId, assignedName }` | Updated task |
| `unassign` | `{ taskId }` | Updated task |

## `POST /api/analytics`

No body. Returns 30-day workspace aggregates: totals, rates (correction, unsupported,
needs-review, evidence citation, signals measured), most-hallucinated fields, error categories,
document types, model comparison, risk distribution, daily trend, recent activity.
`hasData: false` on an empty workspace — there is no sample data anywhere.

## `POST /api/batch-job`

| action | Body |
|---|---|
| `create` | `{ files: string[], label? }` → job + items |
| `item` | `{ jobId, itemId, status, documentId?, trustScore?, … }` |
| `status` | `{ jobId }` |
| `cancel` | `{ jobId }` |

The browser drives the loop and reports each outcome; the job record lives server-side so it
survives a refresh. Max 1000 documents per job.

## `POST /api/benchmark`

```jsonc
{ "image": "data:…", "fileName": "invoice.pdf",
  "upstreamClaims": [...], "models": ["gemini-2.5-flash-lite","gemini-2.5-flash"] }
```

Runs each model through the identical production pipeline, sequentially. Returns per-model
decisiveness, corrections raised, evidence citation rate, signals measured, blocks transcribed and
latency — plus a `disclaimer`: without labelled ground truth these are behavioural measurements,
not accuracy. Limited to 3 runs per 5 minutes, 4 models per run.

## `POST /api/workspace`

| action | Purpose |
|---|---|
| `status` | Workspace, provider registry, benchmark targets, compliance posture |
| `settings` | `{ name?, retentionDays? }` — retention re-applies to stored documents |
| `activity` | API activity log + human review decisions |
| `purge` | Delete documents past retention |
| `erase` | Delete everything in the workspace |

## `GET /api/health`

No workspace needed. For uptime monitors.

```jsonc
{ "status": "healthy | degraded | unhealthy",
  "checks": { "verification": { "ok": true }, "persistence": { "ok": true, "mode": "workspace" } },
  "activeModel": "gemini-2.5-flash", "version": "a1b2c3d" }
```

`503` only when verification cannot run at all. `degraded` means verification works but storage
does not — a monitor should page on `unhealthy`, warn on `degraded`.
