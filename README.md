# TruthLens AI — Enterprise AI Hallucination Verification Platform

> **"Can You Trust What AI Sees?"** — the verification layer between a vision model and the human who acts on its output.

Vision language models hallucinate. A model reading an invoice may report the vendor as
"Microsoft" when the page says "Oracle". TruthLens answers one question per claim: **does the
document actually say this?** — and refuses to answer when it cannot prove it.

It works on any document type with **zero hardcoded fields** anywhere in the frontend or backend,
and it never substitutes sample results when a provider or retrieval fails.

**Measured on a real invoice with planted errors:**

```
CORRECTED    Vendor          → ORACLE CORPORATION   (claim said Microsoft)
VERIFIED     Invoice Number  93%
VERIFIED     Total           93%
CORRECTED    Payment Terms   → Net 30               (claim said Net 90)
UNSUPPORTED  Shipping Weight 0%, HIGH risk          (absent from the document)

5/5 verdicts correct · 4 cited / 14 retrieved evidence · 6 relations derived
```

### Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Why two passes, the guardrails, the trust signals, the layer map |
| [docs/API.md](docs/API.md) | Every endpoint, request/response shapes, error contract |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Both deployment modes, env vars, health checks, rollback, security checklist |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the five invariants, how to add a provider |

### Quick start

```bash
npm install
cp .env.example .env      # add GEMINI_API_KEY from https://aistudio.google.com/apikey
npx vercel dev            # http://localhost:3000
npm run verify            # typecheck + lint + 64 tests + build
```

> **Free-tier quota:** `gemini-2.5-flash` allows 20 requests/day, and verification makes two model
> calls — roughly **10 verifications/day**. Measured, not assumed. Enable billing before demoing.

---

## 🌟 Key Platform Features

### 1. Zero Hardcoding Universal Claims Architecture
The platform is document-agnostic. No document type, field name, or schema is hardcoded anywhere
in the frontend or backend — the UI renders whatever `claims[]` the server returns. It has been
exercised on resumes, contracts, medical reports, purchase orders, and architecture diagrams, but
nothing in the code is specific to any of them.

**What TruthLens verifies.** Two modes:

- **Cross-check (primary).** You supply the claims your AI system produced about a document, and
  TruthLens checks each against evidence in that document. This is the strong mode — the proposer
  and the verifier are different systems.
- **Self-check.** TruthLens extracts the document's business facts as atomic claims, then runs
  them through the identical verification pipeline. Useful when you have no upstream system to
  check, but weaker: the proposer and the verifier share a failure mode, so a fact misread
  identically twice will still verify cleanly. The UI labels every self-check run as such.

### 2. Verification Pipeline (as implemented)

Every stage below actually runs and reports its **measured** duration in the response
`timeline[]`. Nothing is simulated client-side.

```
      [ Upload document + upstream AI claims ]
                        │
                        ▼
   [ 0. Claim extraction ]  api/extract-claims.ts  (self-check only)
        Business facts → atomic claims, editable before verifying
                        │
                        ▼
   [ 1. Intake ]  validate payload, size, claim syntax
                        │
                        ▼
   [ 2. Document understanding ]  api/_documentIndex.ts
        Claim-blind transcription → normalised, deduped,
        junk-filtered block index with page, region,
        coordinates and legibility per block
                        │
                        ▼
   [ 3. Evidence retrieval ]  api/_retrieval.ts
        Independent search — no model in the loop.
        6 strategies: value-match · lexical · numeric ·
        field-label · region-affinity · spatial-neighbour
                        │
                        ▼
   [ 4. Risk prediction ]  api/_signals.ts
        Hallucination risk from retrieval strength and page
        legibility, BEFORE the verifier runs
                        │
                        ▼
   [ 5. Verification ]  provider adapter
        Model may only cite ids the retrieval engine returned
                        │
                        ▼
   [ 6. Reflection & trust scoring ]  5 independent signals;
        unmeasured signals excluded, not substituted;
        ungrounded corrections dropped
                        │
                        ▼
   [ 7. Persistence ]  document → claims → evidence →
        relations → audit trail → review tasks (workspace mode)
                        │
                        ▼
   [ Decision: Verified │ Corrected │ Unsupported │ Needs Review ]
```

**Why two passes.** Transcription never sees a claim. A model asked to "find evidence for X"
will find something resembling X — when the same model both asserts a fact and supplies its own
proof, verification is circular. Retrieval happens between the two passes and the verifier can
only cite what it returned, so evidence is retrieved rather than manufactured to fit.

**Structural guardrails** (enforced in code, not asked for in the prompt):
- A cited id the retrieval engine never returned is discarded.
- `verified`/`corrected` with no resolvable evidence is downgraded to `needs_review`.
- A corrected value that appears in no cited block is dropped.
- Container junk (PDF internals, xref, TeX producer strings) is rejected at index time.

**Known gaps — deliberately not claimed as built:**
- Transcription is a model call, not a dedicated OCR engine. It is claim-blind, which removes the
  circularity, but a wrong transcription is still a wrong index.
- Only the Gemini provider adapter exists. The benchmark therefore compares Gemini models rather
  than vendors, and labels itself as exactly that.
- No labelled ground truth, so the benchmark reports behavioural measurements (decisiveness,
  corrections raised, evidence citation, latency) and explicitly does not claim accuracy.
- Batch sequencing is browser-driven; there is no worker or message broker.
- No authentication, by design — see the security trade-off below.

### 3. Verification Card (`VerificationCard.tsx`)
Each extracted claim renders as a reusable verification card featuring:
- **Category & Field Name** (e.g. `Education · College / University`)
- **Status Badges**:
  - `Verified`: Evidence explicitly confirms raw AI output.
  - `Corrected`: Evidence conflicts with AI output; engine automatically supplies verified truth.
  - `Unsupported`: No evidence exists in document (prevents hallucination when questions cannot be answered).
  - `Needs Review`: Ambiguous or low confidence.
- **Trust Score Pill**: Percentage trust metric (0–100%).
- **Claim Value Transition**: Original AI Output vs Verified Truth (with strike-through for corrected hallucinations).
- **Explainable AI (XAI) Reason Snippet**: Human-readable narrative detailing *why* the decision was made.
- **Provenance Summary**: Evidence signals count & page number indicator.
- **Human-in-the-Loop Buttons**: `Approve`, `Reject`, `Override` buttons.

### 4. Interactive Inspection Drawer (`VerificationSidePanel.tsx`)
Clicking any `VerificationCard` slides open a deep inspection drawer:
- **Explainable Trust Breakdown**: each of the five signals with its score, **what it was measured
  from**, and a "why this score" narrative. Signals nothing measured are shown as excluded rather
  than filled in with another signal's value.
- **Pre-verification hallucination risk**: predicted from retrieval strength and page legibility
  *before* the verifier runs, with the specific reasons (low legibility, no numeric match,
  ambiguous regions).
- **Evidence Retrieval trace**: which surfaces were searched, which strategies hit, and how many
  candidates the verifier cited versus ignored.
- **Interactive Evidence Viewer**: real pdf.js rendering with page navigation, zoom, zoom-to-evidence,
  and normalised bounding-box overlays that distinguish cited from merely-retrieved regions.
- **Storytelling Flow**: `Original AI Output` → `Retrieved Evidence` → `Verified Output` → `Explanation`.
- **Measured Pipeline Timeline** (`PipelineTimeline.tsx`): real server-side stage durations.
- **Human Feedback Form**: approve / reject / override with reviewer comments. Disabled when the
  run was not persisted, because a decision needs a durable claim record to be auditable.

### 5. Compliance Report & Audit Trail (`AuditTrailDrawer.tsx`, `lib/verificationReport.ts`)
- Decision log for the session: original claim, final value, status, trust, signals measured,
  pre-verification risk, and who decided.
- **Full report (PDF)** opens a self-contained compliance document — document metadata, verification
  summary, measured decision timeline, and a per-claim section with signals, "why this score",
  the reasoning trace, cited evidence text with coordinates, and any human decision. It carries
  its own print stylesheet, so what reaches the printer is a record, not a screenshot of the UI.
- **JSON** (full result) and **CSV** (formula-injection safe) exports.

### 6. Model Abstraction Layer (`api/_providers/`)
`VisionProviderAdapter` is the only thing the pipeline knows about. Adding a provider means
implementing `transcribe` + `verify` in one new file and adding one line to the registry — the
retrieval engine, scoring, persistence and UI are untouched.

- **Gemini** (Google DeepMind) — adapter implemented and configured
- Claude · GPT-4o · Llama Vision · Qwen VL — registered in the interface, **no adapter yet**;
  requests naming them are rejected with `501` rather than silently falling back to Gemini.

### 7. Claim Relation Graph (`ClaimRelationGraph.tsx`)
Relationships are derived from where each claim's evidence physically sits — shared evidence
block, shared page region, shared page, or related field name. Nothing here knows what an invoice
or a resume is.

### 8. Human Review Queue (`/review`)
Claims the engine would not decide automatically, across every document in the workspace, with
triage context (pre-verification risk, evidence cited vs retrieved, trust score) and assignment.
This closes the spec's chain — Needs Review → Assign → Comment → Decide → Audit → Final decision.
The decision itself still happens in the claim drawer, where the evidence is; approving, rejecting
or overriding resolves the task and writes to the audit trail automatically.

Reviewer names are self-declared. With no accounts, the audit trail records who *said* they made a
decision, not a verified identity — stated on the page where the name is entered.

### 9. Enterprise Dashboard (`/dashboard`)
Every figure is computed from verifications this workspace actually ran — there is no sample data
anywhere. Most-hallucinated fields, correction / unsupported / needs-review rates, average signals
measured, evidence citation rate, pre-verification risk distribution, top document types, model
comparison, a 30-day trust trend, and recent activity. An empty workspace shows an empty state.

Charts use **labelled rows rather than stacked segments or pie slices**: the app's status palette
was validated for colour-vision separation and warning↔success sits at ΔE 7.5 (protan), inside the
6–8 floor band that is only usable with secondary encoding. Rather than repaint a locked theme,
every value carries its own text label and number, so identity is never colour-alone.

### 10. Batch Verification (`/batch`)
Verify one claim set across many documents. The job record lives server-side (so it survives a
refresh and yields one consolidated report) while the browser sequences submissions — this
deployment has no worker or message broker, and pretending otherwise would be dishonest. Live
per-document progress, failure detail, stop-after-current, and a consolidated JSON export.

### 11. Compliance & Admin (`/admin`)
Workspace name and retention policy (retention re-applies to already-stored documents, not just
future ones), workspace key management with export/switch/forget, apply-retention-now, full
erasure, the API activity log, human review decisions, the live provider registry, the
verification rules actually enforced in code, and a compliance posture table.

The compliance table reports **controls it can verify about this deployment** — encryption in
transit and at rest, retention enforcement, erasure, audit coverage, tenant isolation — and marks
authentication and RBAC as *not provided*. It does not claim SOC 2, GDPR or HIPAA readiness:
those are organisational certifications covering people, contracts and process, and cannot be
asserted by software about itself.

---

## 🛠️ Technology Stack

- **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Framer Motion, Lucide Icons, Recharts.
- **Backend Options**:
  - **Vercel Serverless Functions** (`/api/*` in Node.js/TypeScript).
  - **Python FastAPI Server** (`/python/server.py` with Uvicorn).
- **Database & Persistence**: Supabase PostgreSQL with Row-Level Security (RLS). Optional — see
  the two deployment modes below.
- **AI Models**: Google Gemini VLM API, MediaPipe WASM, COCO-SSD TensorFlow.js.

---

## 🚀 Getting Started & Local Setup

### 1. Prerequisites
- Node.js (v18+ recommended) or Bun
- Python 3.10+ (if running Python backend)
- Google Gemini API Key (Optional — a free key can be obtained at [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Environment Setup

**There is no sign-up.** Anyone can use TruthLens immediately. Tenancy is carried by an anonymous
*workspace token* the browser mints on first use; the server stores only its SHA-256 hash and
scopes every query by the workspace it resolves to.

> **Security trade-off, stated plainly.** The workspace token is a bearer secret, like an unlisted
> share link: whoever holds it has the workspace, there is no second factor, and it cannot be
> recovered — we hold no identifier that could prove it was yours. This is right for open
> self-serve use and is **not** sufficient for regulated personal or health data. The Admin page
> says the same thing to the user, and the Compliance tab lists authentication as *not provided*.

Two modes, decided by environment and never by the client:

| | **Stateless** (default) | **Workspace** |
|---|---|---|
| Trigger | `SUPABASE_*` unset | `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` set |
| Verification | Works for anyone, rate-limited | Works for anyone, rate-limited |
| Storage | Nothing persisted | Auto-provisioned workspace on first request |
| Review / analytics / batch / audit | Unavailable, with the reason shown per feature | Enabled |

Because nothing authenticates, the browser never reaches the database: every table is revoked from
the `anon` and `authenticated` roles, and all access goes through the API, which holds the
service-role key. RLS is defence in depth rather than the access-control mechanism.

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
VITE_BACKEND=api

# Optional — switches the API into workspace mode (apply supabase/migrations first)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

**Rate limiting is in-memory and per function instance.** It stops one caller hammering a warm
instance; it is not a billing control. Put a durable limiter or API gateway in front before
exposing this publicly.

### 3. Installation
Install frontend dependencies:
```bash
npm install
```

If using the Python FastAPI backend:
```bash
cd python
pip install -r requirements.txt
```

### 4. Running the Application locally

#### Option A: React Frontend + Vercel Serverless Functions (Default)
Run the Vite development server:
```bash
npm run dev
```
Open your browser at `http://localhost:8080` or `http://localhost:5173`.

#### Option B: React Frontend + Python FastAPI Backend
Start Python FastAPI backend in terminal 1:
```bash
cd python
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Set `VITE_BACKEND=python` in `.env`, then start React frontend in terminal 2:
```bash
npm run dev
```

### 5. Checks
```bash
npm run typecheck   # tsconfig.app.json (src) + tsconfig.api.json (api, strict)
npm run lint
npm run build
```

---

## 📂 Project Architecture & Directory Layout

```
RLVO-realLOD-main/
├── api/                             # Serverless API Endpoints (strict TS, see tsconfig.api.json)
│   ├── _providers/                  # ── Model abstraction layer ──
│   │   ├── types.ts                 #    VisionProviderAdapter interface
│   │   ├── gemini.ts                #    Gemini adapter (transcribe + verify)
│   │   └── index.ts                 #    Registry — add a provider in one line
│   ├── _geometry.ts                 # Bounding-box normalisation (0-1 / 0-100 / 0-1000 → %)
│   ├── _documentIndex.ts            # Searchable block index from claim-blind transcription
│   ├── _retrieval.ts                # Evidence Retrieval Engine (6 strategies, no model)
│   ├── _signals.ts                  # Independent trust signals, risk prediction, claim graph
│   ├── _truthlens.ts                # Claim assembly, guardrails, stage recorder
│   ├── _gemini.ts                   # HTTP client: timeouts, retries, tolerant JSON parsing
│   ├── _auth.ts                     # Identity, org resolution, ownership checks
│   ├── _ratelimit.ts                # Per-instance rate limiting (guardrail, not a control)
│   ├── _persistence.ts              # Durable write: document → claims → evidence → audit
│   ├── _pipeline.ts                 # Shared pipeline — verify and benchmark run identical code
│   ├── _workspace.ts                # Anonymous workspace resolution, activity log, erasure
│   ├── verify-document.ts           # Verification handler (maxDuration 60s, two provider calls)
│   ├── review-claim.ts              # Human decision handler (workspace-scoped)
│   ├── extract-claims.ts            # Business fact → atomic claim extraction (self-check)
│   ├── review-queue.ts              # Cross-document review queue & assignment
│   ├── analytics.ts                 # Dashboard aggregates, computed from stored runs
│   ├── batch-job.ts                 # Batch job create / progress / cancel
│   ├── benchmark.ts                 # Multi-model head-to-head on one document
│   └── workspace.ts                 # Settings, compliance posture, activity, purge, erase
├── python/                          # Python Backend Implementation
│   ├── server.py                    # FastAPI Web Server
│   └── image_refinement.py          # TruthLens Document Verification Engine
├── src/
│   ├── components/truthlens/        # Enterprise UI Components
│   │   ├── VerificationCard.tsx     # Dynamic Claim Card with XAI & Feedback
│   │   ├── VerificationSidePanel.tsx# Evidence, reasoning, comparison & review drawer
│   │   ├── DocumentEvidenceViewer.tsx # pdf.js viewer: page nav, zoom-to-evidence, overlays
│   │   ├── TrustScoreBreakdown.tsx  # Per-signal score, its basis, and "why this score"
│   │   ├── ClaimRelationGraph.tsx   # Relationships derived from evidence location
│   │   ├── PipelineTimeline.tsx     # Measured server-side stage durations
│   │   ├── charts.tsx               # Labelled-row bars & single-series trend (CVD-safe)
│   │   ├── AuditTrailDrawer.tsx     # Compliance Audit Log & JSON/CSV Exporter
│   │   ├── VisionProviderSelector.tsx# Multi-LLM Provider Selector
│   │   ├── VerificationSummaryCards.tsx # Metrics Cards (Trust Score, Risk)
│   │   └── LandingSections.tsx      # High-Impact Hero & Feature Sections
│   ├── pages/
│   │   ├── TruthLensVerify.tsx      # Main Verification Studio Page
│   │   ├── TruthLensBatch.tsx       # Batch verification with live job progress
│   │   ├── TruthLensReview.tsx      # Human review queue with assignment
│   │   ├── TruthLensBenchmark.tsx   # Multi-model comparison & per-claim disagreement
│   │   ├── TruthLensAdmin.tsx       # Workspace, compliance posture, activity log
│   │   ├── TruthLensDashboard.tsx   # Analytics & Risk Dashboard Page
│   │   ├── ImageRefinement.tsx      # Real-LOD Image Captioning Re-alignment
│   │   └── Proctoring.tsx           # Verification-First Live Proctoring
│   ├── types/
│   │   └── truthlens.ts             # TruthLens v2.0 Enterprise TypeScript Interfaces
│   └── lib/
│       ├── workspace.ts             # Anonymous workspace token (mint / switch / forget)
│       ├── verificationReport.ts    # Compliance report, JSON & CSV export
│       └── visionProviders.ts       # Provider display metadata
├── supabase/
│   └── migrations/                  # PostgreSQL Schema Migration Scripts
│       ├── 20260804000000_universal_claims_schema.sql
│       ├── 20260805000000_fix_rls_policy_lockout.sql
│       ├── 20260806000000_retrieval_provenance.sql
│       └── 20260807000000_anonymous_workspaces.sql     # Apply ALL FOUR, in order
├── package.json
└── vite.config.ts
```

---

## 🗄️ Database Schema (`supabase/migrations/`)

Apply **all four** migrations, in filename order. The second is not optional: the first enables
RLS on `claim_evidence`, `verification_timeline` and `audit_trails` while dropping their policies,
which locks those tables out entirely for non-service clients.

- `organizations` / `organization_members`: tenancy and roles (`admin`, `reviewer`, `member`, `viewer`).
- `documents`: document metadata, trust score, risk level, retention window, processing status.
- `claims`: verified claims, status, the five-signal breakdown, how many signals were measured,
  pre-verification risk, and the stored score rationale (so a report regenerates identically).
- `claim_evidence`: transcribed text, page numbers, normalised bounding boxes, which retrieval
  strategies found it, and whether the verifier cited it or ignored it.
- `claim_relations`: derived relationships between claims (shared evidence / region / page).
- `organizations`: anonymous workspaces, keyed by the SHA-256 hash of the workspace token.
- `verification_jobs` / `verification_job_items`: batch jobs and their per-document outcomes.
- `model_benchmarks`: benchmark runs, one row per model per run.
- `api_activity`: route, action, status and duration for the compliance activity log.
- `verification_timeline`: the measured stage durations for each run.
- `audit_trails`: append-only record of every decision — machine and human.
- `review_tasks` / `review_decisions`: the human-review queue and its outcomes.


---

## ⚖️ License & Attribution

Inspired by the **Real-LOD (ICLR 2025)** research workflow for agentic vision-language grounding and verification.
