# TruthLens AI — Enterprise AI Trust Layer

> **"Can You Trust What AI Sees?"** — the verification layer between a vision model and the human who acts on its output.

Vision language models hallucinate. A model reading an invoice may report the vendor as
"Microsoft" when the page says "Oracle". TruthLens answers one question per claim: **does the
document actually say this?** — and refuses to answer when it cannot prove it.

It works on any document type with **zero hardcoded fields** anywhere in the frontend or backend,
and it never substitutes sample results when a provider or retrieval fails.

**Measured on a real invoice with planted errors:**

```
CORRECTED    Vendor           →  Oracle Corporation    (claim said Google)
VERIFIED     Invoice Number   94%
CORRECTED    Total            →  $13,511.00            (claim said $13,51.00)
CORRECTED    Payment Terms    →  Net 30                (claim said Net 99)
UNSUPPORTED  Shipping Weight  50%                       (absent from the document)
```

---

## The two verification modes

**Cross-check — the primary path.**

```
Document  →  External AI (ChatGPT / Claude / Gemini / any vision LLM)  →  Generated claims
                                                                              ↓
                                                        TruthLens Verification Engine
                                                                              ↓
                                                                     Verified results
```

You supply the claims your AI produced. TruthLens **never trusts them**: it independently extracts
the document's text, retrieves evidence with a deterministic search engine, and only then asks a
model to judge each claim against what was found.

**Self-check — the fallback.** With no external AI to hand, TruthLens proposes the document's facts
itself and runs them through the identical pipeline. Weaker evidence, because the proposer and the
verifier share a failure mode — the UI labels every such run **Self-check mode** and says why.

The mode is shown before you type a claim and again on the result. It is never inferred silently.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Why OCR is deterministic, the guardrails, the trust signals, the layer map |
| [docs/API.md](docs/API.md) | Every endpoint, request/response shapes, error contract |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Both modes, auth setup, the OCR service, env vars, rollback |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the invariants, how to add a provider |

## Quick start

```bash
npm install
cp .env.example .env      # add GEMINI_API_KEY or OPENROUTER_API_KEY
npx vercel dev            # http://localhost:3000
npm run verify            # typecheck + lint + 76 tests + build
```

Works immediately as a guest, with no account and no database.

---

## Pipeline

Every stage below actually runs and reports its **measured** duration in `timeline[]`. Nothing is
simulated client-side.

```
      [ Upload: PDF · PNG · JPG · JPEG · WEBP · TIFF ]
                        │
                        ▼
   [ 0. Intake ]  src/lib/documentInput.ts
        PDFs rasterised to page images IN THE BROWSER, so the
        engine only ever handles images — one input shape, no
        vendor-specific PDF handling anywhere downstream
                        │
                        ▼
   [ 1. Text extraction ]  api/_ocr.ts → python/ocr_service.py
        PaddleOCR: text, bounding boxes, per-word confidence, page.
        DETERMINISTIC — the same page always yields the same text
        and coordinates, and costs no model quota
                        │
                        ▼
   [ 2. Evidence retrieval ]  api/_retrieval.ts
        No model in the loop. 6 strategies: value-match · lexical ·
        numeric · field-label · region-affinity · spatial-neighbour
                        │
                        ▼
   [ 3. Risk prediction ]  api/_signals.ts
        Hallucination risk from retrieval strength and legibility,
        BEFORE any verdict exists
                        │
                        ▼
   [ 4. Verification ]  provider adapter — THE ONLY REASONING STEP
        The model may only cite evidence ids retrieval returned
                        │
                        ▼
   [ 5. Reflection & trust scoring ]  5 independent signals;
        unmeasured signals excluded, not substituted;
        ungrounded corrections dropped
                        │
                        ▼
   [ 6. Persistence ]  scoped to the signed-in user (workspace mode)
                        │
                        ▼
   [ Verified │ Corrected │ Unsupported │ Needs Human Review ]
```

**AI is used in exactly one place: step 4.** Reading, retrieval, risk, scoring, persistence and
analytics are all deterministic code. That is what makes results reproducible and cheap.

### Structural guardrails

Enforced in code, never merely requested in a prompt:

| Guardrail | Where |
|---|---|
| A cited id retrieval never returned is discarded | `_truthlens.ts` |
| `verified`/`corrected` with no resolvable evidence → `needs_review` | `_truthlens.ts` |
| A corrected value absent from cited text is dropped | `_truthlens.ts` |
| Container junk (PDF internals, xref, TeX) rejected at index time | `_documentIndex.ts` |
| Unmeasured trust signals excluded from the mean, never substituted | `_signals.ts` |

---

## Guest vs signed-in

| | **Demo mode** (guest) | **Workspace mode** (signed in) |
|---|---|---|
| Verification | ✅ Full pipeline | ✅ Full pipeline |
| Results stored | ❌ | ✅ Scoped to your account |
| Dashboard | ❌ | ✅ Real history only |
| Review queue | ❌ | ✅ |
| Audit trail | ❌ | ✅ Verified reviewer identity |

Sign-in is **Google via Supabase Auth**. Every review decision records the reviewer's name and
email from the verified session — never a typed-in string, because a self-declared reviewer field
is not an audit trail.

Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to enable it. Without them the app runs
entirely in demo mode and says so.

---

## Model providers

`VisionProviderAdapter` is the only thing the pipeline knows about. Adding a vendor means one new
file plus one registry line.

| Provider | Models |
|---|---|
| **Gemini** (direct) | `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest` |
| **OpenRouter** | `anthropic/claude-sonnet-4`, `openai/gpt-4.1`, `google/gemini-2.5-flash`, `qwen/qwen2.5-vl-72b-instruct` |

Switch provider and model from **Admin → Models** — no code change, no redeploy. If the chosen
provider fails (quota, outage, retired model), TruthLens automatically retries with the next
configured provider **inside the same request** and reports which one produced the result.

Models are chosen for a generous free tier, strong vision, and a stable API. Pinned versions are
avoided: Google retires them for new projects, which 404s a fresh key.

---

## Interactive evidence viewer

Clicking a claim opens the drawer, navigates to the evidence page, and highlights the exact region:

- Real **pdf.js** rendering, whole-page fit, zoom, page navigation
- **Zoom-to-evidence** and synchronised selection between the list and the overlay
- Bounding boxes aligned to OCR coordinates — verified in a headless browser by asserting
  `(box.left − canvas.left) / canvas.width × 100` equals the evidence `boundingBox.x`
- Cited evidence is visually distinct from retrieved-but-unused

Every value in the drawer comes from the verification response. Nothing is generated for display.

---

## Verification & testing

```bash
npm run verify        # typecheck (src + api strict) → lint → 76 tests → build
npm test              # unit + regression suites
```

| Layer | How verified |
|---|---|
| Migrations (5) | Applied to real PostgreSQL 16 in Docker; idempotent on re-run |
| User-scoped data flow | Insert → review → retention → purge → cascade, all asserted |
| Pipeline logic | 76 tests: geometry, retrieval, signals, guardrails, media, security |
| Verification correctness | Live run against a real invoice with planted errors — 5/5 |
| Evidence overlay | Headless browser, box offsets measured against canvas |

---

## Folder structure

```
api/
├── _ocr.ts              # OCR engine abstraction; PaddleOCR primary, model fallback
├── _documentIndex.ts    # Searchable block index built from OCR output
├── _retrieval.ts        # Evidence Retrieval Engine — deterministic, no model
├── _signals.ts          # Trust signals, risk prediction, claim relation graph
├── _truthlens.ts        # Claim assembly and the decision guardrails
├── _pipeline.ts         # The sequence, shared by verification and benchmark
├── _identity.ts         # Supabase session verification, ownership checks, activity log
├── _media.ts            # Magic-byte format detection
├── _persistence.ts      # Durable write: document → claims → evidence → audit
├── _providers/          # Model abstraction: gemini.ts, openrouter.ts, index.ts
├── verify-document.ts · extract-claims.ts · review-claim.ts · review-queue.ts
├── analytics.ts · benchmark.ts · workspace.ts · health.ts
python/
└── ocr_service.py       # PaddleOCR FastAPI service (separate host — see DEPLOYMENT)
src/
├── integrations/auth.tsx      # Google sign-in, demo/workspace mode
├── lib/documentInput.ts       # PDF → page images in the browser
├── components/truthlens/      # Presentation; no component knows any document type
└── pages/                     # One page per route, rendering backend JSON only
supabase/migrations/           # Apply ALL FIVE, in filename order
```

---

## Known limitations

Stated plainly rather than left for you to discover:

- **PaddleOCR needs a separate host.** It is a Python library and cannot run inside Vercel's Node
  functions. Without `OCR_SERVICE_URL` the pipeline falls back to model transcription and labels
  the run `model OCR` instead of `PaddleOCR`.
- **Free-tier quota.** `gemini-flash-*` allows 20 requests/day. Enable billing before demoing.
- **OpenRouter PDFs** need a paid balance for their file-parser; images work on a free balance.
  TruthLens rasterises PDFs client-side, so this rarely bites.
- **Batch verification is removed** from the product surface. The API and page remain in the tree
  so it can return; no dead route is exposed.
- **No organisation roles.** Each account sees only its own data; there is no team sharing yet.
- **Benchmark reports behaviour, not accuracy** — there is no labelled ground truth, and calling
  decisiveness "accuracy" would be a fabrication.

---

## License & attribution

Inspired by the **Real-LOD (ICLR 2025)** research workflow for agentic vision-language grounding
and verification.
