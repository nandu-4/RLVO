# TruthLens AI — Architecture

## The problem this solves

Vision language models hallucinate. A model reading an invoice may report the vendor as
"Microsoft" when the page says "Oracle". TruthLens sits between such a model and the human who
acts on its output, and answers one question per claim: **does the document actually say this?**

## Why two passes

The naive design — ask one model to verify a claim and supply its own evidence — is circular. A
model asked to "find evidence for X" will find something resembling X, and a fact it misread will
come back with confidently misread evidence attached.

TruthLens splits the work:

```
┌──────────────────────────────────────────────────────────────────┐
│ PASS 1 · Document understanding          api/_providers/*        │
│ Transcribe every visible block. The model NEVER sees a claim.    │
│ Output: text, page, region, coordinates, legibility per block.   │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Document index                           api/_documentIndex.ts   │
│ Normalise coordinates, reject container junk, dedupe, tokenise.  │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Evidence Retrieval Engine                api/_retrieval.ts       │
│ NO MODEL IN THE LOOP. Six strategies rank candidate blocks:      │
│ value-match · lexical · numeric · field-label · region-affinity  │
│ · spatial-neighbour                                              │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Hallucination risk prediction            api/_signals.ts         │
│ From retrieval strength + page legibility, BEFORE verification.  │
│ Independent of the outcome, therefore actionable in advance.     │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ PASS 2 · Verification                    api/_providers/*        │
│ Model sees the claim + the retrieved candidates. It may only     │
│ CITE those candidates by id. It cannot mint evidence.            │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Reflection & trust scoring          api/_signals.ts, _truthlens  │
│ Five independently measured signals; unmeasured ones excluded.   │
│ Guardrails applied here, not trusted to the prompt.              │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
        Verified · Corrected · Unsupported · Needs Human Review
```

`api/_pipeline.ts` owns this sequence, and both `verify-document` and `benchmark` call it — a
benchmark that measured different code than production would be worse than no benchmark.

## Structural guardrails

Enforced in code, never merely requested in a prompt:

| Guardrail | Where | Why |
|---|---|---|
| Cited id must exist in retrieval output | `_truthlens.ts` `assembleClaim` | Stops invented citations |
| `verified`/`corrected` with no resolvable evidence → `needs_review` | `_truthlens.ts` `decideStatus` | An undecidable claim is not a decision |
| Correction must appear verbatim in cited text | `_truthlens.ts` `resolveVerifiedValue` | Stops invented corrections |
| Container junk rejected at index time | `_documentIndex.ts` | PDF internals are not business facts |
| Unmeasured signal excluded from the mean | `_signals.ts` `computeSignals` | Five copies of one number is not corroboration |

## Trust signals

Each is measured from a different source, so agreement between them is real corroboration:

| Signal | Measured from |
|---|---|
| OCR agreement | Character-level similarity of claim vs transcribed text, weighted by legibility |
| Semantic agreement | Content-token overlap between claim and cited evidence |
| Layout agreement | Coordinate coverage, region fit, pairwise spatial coherence |
| Vision agreement | The provider's visual read — the only model-supplied signal, and the only one that may legitimately be missing |
| Evidence strength | Retrieval ranking and legibility of what the engine found |

The final score is a weighted mean **over measured signals only**, renormalised. A signal nothing
measured is reported as excluded — never defaulted to another signal's value.

## Tenancy without accounts

There is no sign-up. The browser mints a 256-bit workspace token; the server stores only its
SHA-256 hash and scopes every query by the workspace it resolves to.

The token is a bearer secret — like an unlisted share link. Whoever holds it has the workspace,
there is no second factor, and it cannot be recovered. Appropriate for open self-serve use;
**not** sufficient for regulated personal or health data.

Because nothing authenticates, the browser never reaches the database: every table is revoked from
the `anon` and `authenticated` roles and all access goes through the API, which holds the
service-role key. RLS is defence in depth, not the access-control mechanism.

## Layers

```
Browser ──► /api/* (Vercel serverless, Node)  ──► Vision provider (adapter)
                        │                          Google Gemini today
                        ▼
              PostgREST ──► PostgreSQL          (optional; stateless without it)
```

| Directory | Responsibility |
|---|---|
| `api/_providers/` | Model abstraction. One file per vendor; the pipeline knows only the interface |
| `api/_documentIndex.ts` | Searchable block index built from claim-blind transcription |
| `api/_retrieval.ts` | Evidence Retrieval Engine — deterministic, no model |
| `api/_signals.ts` | Trust signals, risk prediction, claim relation graph |
| `api/_truthlens.ts` | Claim assembly and the decision guardrails |
| `api/_pipeline.ts` | The sequence, shared by verification and benchmark |
| `api/_workspace.ts` | Workspace resolution, ownership checks, activity log |
| `api/_persistence.ts` | Durable write: document → claims → evidence → relations → audit |
| `src/pages/` | One page per route; all render backend JSON only |
| `src/components/truthlens/` | Presentation. No component knows any document type |

## Adding a provider

1. Implement `VisionProviderAdapter` (`transcribe`, `verify`, `extractClaims`) in
   `api/_providers/<vendor>.ts`.
2. Add one line to the registry in `api/_providers/index.ts`.

Nothing else changes — retrieval, scoring, persistence and the UI all work against the interface.

## Known architectural limits

- Transcription is a model call, not a dedicated OCR engine. It is claim-blind, which removes the
  circularity, but a misread page still yields a misread index.
- Batch sequencing is browser-driven; there is no worker or message broker.
- The rate limiter is in-memory and therefore per function instance.
- No authentication, by design.
