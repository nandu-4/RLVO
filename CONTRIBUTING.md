# Contributing to TruthLens AI

## Setup

```bash
npm install
cp .env.example .env      # add GEMINI_API_KEY
npx vercel dev            # frontend + /api on http://localhost:3000
```

## Before every push

```bash
npm run verify            # typecheck (src + api, strict) + lint + tests + build
```

All four must pass. `npm run typecheck` covers both `tsconfig.app.json` (frontend) and
`tsconfig.api.json` (serverless handlers, `strict: true`).

## The rules that matter

These are not style preferences. They are what the product claims to be, and a change that breaks
one of them breaks the product's core promise.

**1 · Never hardcode a document type or field name.**
No `if (documentType === "invoice")`, no `VENDOR_FIELD` constant, no per-type component. The
frontend renders whatever `claims[]` the server returns and knows nothing else. This is the single
most important invariant in the codebase.

**2 · Never fabricate.**
No mock data, no placeholder metrics, no sample rows "to make the page look full". If there is no
data, render an empty state that explains why. If a signal was not measured, mark it unmeasured —
do not substitute another value.

**3 · Never let a model supply its own evidence.**
Transcription must stay claim-blind. Verification may only cite ids the retrieval engine returned.
If you add a provider, preserve this: it is the difference between verification and theatre.

**4 · Enforce guardrails in code, not in prompts.**
A prompt is a request; a model may decline it. Every rule that matters — citation validity,
evidence requirements, grounded corrections — is enforced in `_truthlens.ts` after the model
replies. Add new rules there.

**5 · Claim what is true.**
No unearned compliance badges, no "accuracy" figure without labelled ground truth, no capability
described in the UI that the code does not implement. If something is a known limitation, say so
in the interface where a user would otherwise be misled — not only in the README.

## Adding a vision provider

1. Implement `VisionProviderAdapter` in `api/_providers/<vendor>.ts` — `transcribe`, `verify`,
   `extractClaims`.
2. Add one line to `ADAPTERS` in `api/_providers/index.ts`.

Nothing else should need to change. If it does, the abstraction has leaked — fix that instead.

## Tests

```bash
npm test                  # all suites
npm run test:coverage     # with coverage over api/
```

`tests/` covers the deterministic core: geometry, retrieval, signals, decision guardrails,
error sanitisation. **Every bug fixed in this area needs a regression test** — two shipped bugs
(coordinate corruption and JSON envelope drift) were invisible to typecheck, lint and build, and
were only caught by running the real pipeline. The tests now encode both.

Pure functions are the priority. Provider calls are not unit-tested — they need a live key and
burn daily quota; exercise them manually against a real document instead.

### Visual verification of the evidence viewer

Two shipped bugs in the overlay were invisible to every automated check and were only found by
rendering the component in a real browser: coordinates corrupted by scale inference, and boxes
positioned against the scroll container instead of the page. If you change
`DocumentEvidenceViewer.tsx`, render it and measure:

```bash
npx playwright install chromium
# mount the component in a scratch route, point a browser at it, then assert that each
# overlay's offset from the canvas equals its boundingBox percentage.
```

The check that matters: `(box.left - canvas.left) / canvas.width * 100` must equal the evidence
`boundingBox.x`. If it does not, the positioning context is wrong.

## Database changes

Migrations live in `supabase/migrations/` and must be:

- **idempotent** — every `CREATE POLICY` preceded by `DROP POLICY IF EXISTS`, every column added
  with `IF NOT EXISTS`. Verify by applying twice.
- **additive** — new tables and nullable columns only, so an application rollback stays safe.
- **verified against real PostgreSQL**, not assumed:

```bash
docker run -d --name tl-pg -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=truthlens -p 55432:5432 postgres:16-alpine
for f in supabase/migrations/*.sql; do docker exec -i tl-pg psql -U postgres -d truthlens -v ON_ERROR_STOP=1 < "$f"; done
```

## Accessibility

Non-negotiable for any UI change:

- Every interactive element reachable by keyboard, with a visible `:focus-visible` ring.
- Icon-only buttons carry `aria-label`.
- Colour is never the sole carrier of meaning — status always has a text label beside it.
- New animation respects `prefers-reduced-motion`; JS-driven animation must check
  `useReducedMotion()` because CSS cannot stop a canvas loop.

## Commit style

Explain **why**, not what — the diff already shows what. A comment or commit message that says
"fixed bug" is worth nothing; one that says "one drifting axis rescaled the whole box, corrupting
21 of 24 overlays" saves the next person an afternoon.
