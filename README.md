# TruthLens AI — Enterprise AI Trust Layer

> **"Can You Trust What AI Sees?"** — the verification layer between a vision model and the human who acts on its output.

Vision language models hallucinate. A model reading an invoice may report the vendor as
"Microsoft" when the page says "Oracle". TruthLens answers one question per claim: **does the
document actually say this?** — and refuses to answer when it cannot prove it.

It works on any document type with **zero hardcoded fields** anywhere in the frontend or backend,
and it never substitutes sample results when a provider or retrieval fails.

The same principle — *never trust a vision model's first answer; check it against the pixels* —
also drives three **RLVO research tools** shipped under **Labs**: **Image RLVO**, **Video RLVO**
and **Proctoring**. See [RLVO research tools](#rlvo-research-tools).

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
npm run verify            # typecheck + lint + 99 tests + build
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

## Storage, history and no-cost replay

**Results are stored by default. Nothing needs configuring for that to be true.**

Every successful verification saves its **complete response** — not a verdict summary. The stored
snapshot carries document metadata, the OCR blocks the reading was derived from, retrieved evidence
with page coordinates, per-claim reasoning and trust breakdowns, the claim relation graph, risk
prediction, measured stage timings, provider, model, failover attempts and timestamp. A replay
returns that object verbatim, so a replayed session **is** the original run rather than a
reconstruction of it.

Two things follow, and both cost zero API calls:

- **Replay** — open anything in **History** to re-examine it. No provider is contacted.
- **Repeat detection** — re-verifying the same document with the same claims returns the stored
  result instead of paying for an answer already known. Send `force: true` to re-run deliberately;
  explicitly choosing a provider or model in Admin → Models also bypasses it, so a model choice is
  never silently ignored.

### The two storage drivers

| | **Local** (default, zero-config) | **Supabase** (set `SUPABASE_URL`) |
|---|---|---|
| Setup | None | Project, service key, migration |
| Verification | ✅ Full pipeline | ✅ Full pipeline |
| Results stored | ✅ Scoped to your browser | ✅ Scoped to your account |
| History + replay | ✅ | ✅ |
| Repeat detection | ✅ | ✅ |
| Dashboard | ❌ | ✅ Real history only |
| Review queue | ❌ | ✅ |
| Audit trail | ❌ | ✅ Verified reviewer identity |

The local driver keys sessions to a browser-generated device id. That is a **scoping key, not a
credential** — it separates one browser's history from another's and nothing more. Review queues
and audit trails stay on Supabase deliberately: they need a verified human, and a device id cannot
supply one. Anything shared, multi-user or regulated wants Supabase.

Local sessions live in `.truthlens-data/` (override with `TRUTHLENS_DATA_DIR`, disable with
`TRUTHLENS_LOCAL_STORE=off`). **On Vercel's managed runtime only `/tmp` is writable and it is
per-instance and evicted freely**, so sessions there last minutes rather than days — `/api/health`
and the History page both report `ephemeral` when that is the case instead of implying permanence.
For durable hosted storage, configure Supabase.

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

# RLVO research tools

TruthLens grew out of three earlier **Real-LOD (RLVO)** experiments, and all three ship in the
app under **Labs** in the navigation. They are not demos bolted on afterwards — they are where the
core idea was worked out: *a vision model's first answer is not trustworthy, so make a second pass
that checks the first against the pixels.* TruthLens applies that to documents. These apply it to
captions, video and live webcam proctoring.

Each has **two implementations that deliberately mirror each other**: a browser/serverless path
used by the app, and a standalone Python reference in `python/` that runs from the command line.
The Python versions exist so the thresholds and prompts can be inspected, tuned and reproduced
without a browser — and any drift between the two is a bug.

---

## 1 · Image RLVO — hallucination injection, then repair

**Route** `/image-refinement` · **Page** [src/pages/ImageRefinement.tsx](src/pages/ImageRefinement.tsx)
· **Reference** [python/image_refinement.py](python/image_refinement.py)
· **Endpoints** `POST /generate-caption`, `POST /refine-caption`

The counter-intuitive part: **stage 1 is engineered to hallucinate on purpose.** You cannot
demonstrate that a repair loop works unless there is real damage to repair, so the first prompt
forbids hedging and *demands* invention — specific brands, exact materials, backstory, emotions,
events outside the frame. It runs at **temperature 1.3** to push the model further into confabulation.

```
Stage 1 · GENERATE     temp 1.3   "Never hedge. Commit to brands, backstory, emotion."
                                   → a fluent, confident, partly fictional 5-7 sentence caption
                        ↓
Stage 2 · VERIFY       temp 0.0   Decompose into ATOMIC claims, judge each against the image
                                   6 aspects: category · attribute · accessory ·
                                              relation · location · behavior
                                   verdict: CORRECT | WRONG | UNCERTAIN (+ correction)
                        ↓
Stage 3 · REWRITE      temp 0.2   Keep CORRECT · apply corrections for UNCERTAIN ·
                                   discard WRONG entirely
```

**Why the temperature swing matters.** Generation at 1.3 maximises invention; verification at
**0.0** makes judgement deterministic and repeatable. Using one temperature for both would either
produce a boring caption with nothing to fix, or a verifier that hallucinates its own verdicts.

**Invented content is always WRONG, by rule.** The verify prompt states that brands, backstory,
emotions and anything outside the frame are wrong *even when plausible* — the test is visibility,
not likelihood. That is the same principle as TruthLens's evidence-required-to-decide guardrail.

**The workflow log is real.** Every `[OK] / [X] / [~]` line is generated from the actual verdict
array returned by stage 2, with counts of kept/dropped/corrected. If the verify pass fails to
parse, the code falls back to a single-pass rewrite and emits a *generic* log instead — the two are
distinguishable, so the UI never shows a fabricated audit of work that did not happen.

| | |
|---|---|
| Model | `gemini-2.5-flash-lite`, `thinkingBudget: 0` |
| Token budgets | generate 500 · verify 1200 · rewrite 400 |
| Retries | 4 attempts, exponential backoff, on `429/500/502/503/504` |

---

## 2 · Video RLVO — temporal sampling

**Route** `/video-refinement` · **Page** [src/pages/VideoRefinement.tsx](src/pages/VideoRefinement.tsx)
· **Reference** [python/video_refinement.py](python/video_refinement.py)
· **Endpoint** `POST /analyze-video`

Vision models take images, not video. Both modes therefore reduce a clip to **evenly-spaced key
frames** (OpenCV `cv2.VideoCapture` seeking by frame index, so sampling is uniform across the whole
clip rather than clustered at the start) and send them as a single multi-image prompt.

| Mode | Frames | Output |
|---|---|---|
| **Summary** | 6 | One 2–3 sentence description of the whole clip |
| **Time capsule** | 8 | One caption per frame — a timeline of what changes |

Time-capsule replies are parsed with a numbered-caption parser that tolerates the model dropping or
merging lines; a short reply yields fewer captions rather than a crash. Same model, retry and
backoff policy as Image RLVO.

---

## 3 · Proctoring — geometric detection with an adversarial AI second opinion

**Route** `/proctoring` · **Page** [src/pages/Proctoring.tsx](src/pages/Proctoring.tsx)
· **Engine** [src/hooks/useProctoring.ts](src/hooks/useProctoring.ts) (~1,000 lines)
· **Reference** [python/proctoring.py](python/proctoring.py) · **Endpoint** `POST /verify-flag`

The most developed of the three, and the clearest expression of the RLVO thesis. A fast geometric
detector raises flags; **a vision model then tries to refute each one before it costs the candidate
anything.**

### What runs in the browser

| Library | Version | Role |
|---|---|---|
| MediaPipe **Face Mesh** | `0.4.1633559619` (jsDelivr CDN) | 468 landmarks + iris refinement |
| **TensorFlow.js** | `4.15.0` | Inference runtime |
| **COCO-SSD** | `2.2.2` | Object detection (phones, novel objects) |

The Python reference swaps COCO-SSD for **Ultralytics YOLOv8n** — same job, same `cell phone`
class, better accuracy off-browser.

### The seven signals

| Signal | Derived from | Severity |
|---|---|---|
| Head turn L/R | nose-to-ear-midpoint offset ÷ half ear span → yaw | medium |
| Gaze off-screen | iris centre (landmarks **468**/**473**) vs eye midpoint | medium |
| Looking down | face aspect ratio compressing vs baseline | **high** |
| Multiple faces | face-mesh count > 1 | **high** |
| No face | absence of landmarks past a 1.5 s grace window | medium |
| Phone in frame | COCO-SSD `cell phone` | **high** |
| Novel object | any new COCO class vs the calibrated room baseline | **high** |

Tab switches are also recorded, from the Page Visibility API rather than the camera.

### Corner calibration — why gaze needs five stages

A fixed gaze threshold fails immediately in practice: it depends on the person's face, their
distance from the camera, and where the webcam physically sits. So the session calibrates against
**the actual screen** before judging anything:

```
60 frames   look at CENTER            → faceAR + gaze baseline
40 × 4      look at each CORNER       → the real horizontal gaze range for THIS screen
            (first 12 frames of each corner discarded while the eyes travel)
```

Gaze is then flagged only outside that measured range plus a **20 % margin**, and a
`GAZE_MIN_RANGE` floor stops a badly-performed calibration from collapsing the range to near-zero
and flagging everything. A 2-frame *leak* tolerance absorbs blinks and tracking jitter, and gaze is
only assessed while the head is centred — a turned head is already its own flag, and scoring both
would penalise one behaviour twice.

### Tuned thresholds

| Constant | Value | Reason |
|---|---|---|
| `HEAD_TURN_THRESHOLD` | `0.33` | Yaw as a fraction of half the ear span |
| `LOOK_DOWN_AR_THRESHOLD` | `0.10` | 10 % face-height compression |
| `LOOK_DOWN_SUSTAINED` | `20` frames (~0.7 s) | Ignores a glance at the keyboard |
| `GAZE_DELTA` / `GAZE_SUSTAINED` | `0.05` / `8` frames | With `GAZE_LEAK = 2` for blinks |
| `PHONE_SCORE_THRESHOLD` | `0.35` | Low on purpose — see below |
| `PHONE_CHECK_MS` | `700` | Detector cadence, at 512 px input |
| `NO_FACE_GRACE_MS` | `1500` | Survives a dropped frame or a stretch |

**Phones alert on first sight, at a deliberately low confidence.** A candidate photographing the
question paper is in and out of frame in about two seconds, so waiting for multi-frame confirmation
means missing it entirely. The threshold is low because a false positive costs only a verification
call — the AI verifier rejects it before any penalty lands. COCO's `remote` class is treated as
phone *suspicion* for the same reason: it is the class COCO most often assigns to a hand-held phone.

**Novel-object detection** records every object COCO sees during calibration as the room's baseline.
Any new class appearing mid-session is flagged instantly and sent to the verifier asking *"is this
an exam-cheating aid?"* — a water bottle or charger is refuted, notes or a second screen are not.
Each class flags once per session, then joins the baseline.

### The adversarial verification loop

This is the part that makes it RLVO rather than an alarm system:

```
detector flags  →  capture the frame at 640 px  →  POST /verify-flag
                                                          ↓
        Gemini, temp 0.0, told: "detectors are FREQUENTLY WRONG.
        A wrong CONFIRMED unfairly accuses a real person.
        Confirm ONLY what you can clearly see."
                                                          ↓
                    CONFIRMED → full penalty
                    REFUTED   → dismissed, no penalty, logged as dismissed
                    UNCERTAIN → half penalty
```

Each flag type carries its **own** question, written to name the innocent explanation explicitly —
the phone question lists chargers, power banks, remotes, calculators, wallets and glasses cases and
instructs the model to answer REFUTED rather than hide behind UNCERTAIN when one of those fits
better. Verification is capped at **2 concurrent calls**; beyond that a flag takes the unverified
penalty rather than queueing, so the detector never stalls behind the network.

### Trust score

Starts at 100 and decays per confirmed flag: **high −6 · medium −3 · low −1 · info 0**, halved for
UNCERTAIN verdicts. Per-type cooldowns (2–5 s) stop one continuous behaviour from draining the
score. The exported session report contains every alert with its timestamp, severity, verdict and
the verifier's evidence sentence — `python proctoring.py` writes the same structure to
`proctoring_report.json`.

---

## Running the RLVO tools standalone

```bash
cd python && pip install -r requirements.txt

python image_refinement.py photo.jpg              # generate → verify → rewrite, prints the log
python video_refinement.py clip.mp4 summary       # or: timecapsule
python proctoring.py                              # webcam;  'q' quits, writes the report
python proctoring.py recording.mp4                # offline analysis of a recorded video
```

They need only `GEMINI_API_KEY`. To point the React app at the Python backend instead of the
serverless functions, set `VITE_BACKEND=python` and run `uvicorn server:app --port 8000`.

---

## Verification & testing

```bash
npm run verify        # typecheck (src + api strict) → lint → 99 tests → build
npm test              # unit + regression suites
```

| Layer | How verified |
|---|---|
| Pipeline logic | **99 tests**: geometry, retrieval, signals, guardrails, media, security, storage |
| Migrations | Applied to real PostgreSQL 16 in Docker, and to a live Supabase project; idempotent |
| Storage + replay | Live round trip — store → list → replay → delete, asserting **0 AI calls** on replay |
| Repeat detection | Identical document + claims served from storage; `force:true` re-runs the model |
| Provider failover | Live: an exhausted model 429s and a sibling model completes the request |
| Verification correctness | Live run against a real invoice with planted errors — 5/5 |
| Evidence overlay | Headless browser, box offsets measured against canvas |
| Tenant isolation | A second device id sees zero sessions; path traversal in the id is rejected |

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
├── _providers/          # Model abstraction: gemini.ts, openrouter.ts, huggingface.ts, index.ts
├── _localstore.ts       # Zero-config file-backed session store
├── _store.ts            # Storage abstraction: supabase | local, one interface
├── verify-document.ts · extract-claims.ts · review-claim.ts · review-queue.ts
├── history.ts           # History listing and no-model replay
├── analytics.ts · benchmark.ts · workspace.ts · health.ts
python/                  # RLVO reference implementations + the OCR service
├── ocr_service.py       # PaddleOCR FastAPI service (separate host — see DEPLOYMENT)
├── image_refinement.py  # Image RLVO: generate → verify → rewrite
├── video_refinement.py  # Video RLVO: frame sampling → summary / time capsule
├── proctoring.py        # Proctoring: MediaPipe Face Mesh + Ultralytics YOLOv8
└── server.py            # FastAPI app exposing all of the above
src/
├── integrations/auth.tsx      # Google sign-in, demo/workspace mode
├── lib/documentInput.ts       # PDF → page images in the browser
├── hooks/useProctoring.ts     # Live proctoring engine + adversarial verification
├── components/truthlens/      # Presentation; no component knows any document type
└── pages/
    ├── ImageRefinement.tsx · VideoRefinement.tsx · Proctoring.tsx   # RLVO Labs
    └── TruthLensVerify.tsx · TruthLensHistory.tsx · ...             # TruthLens
supabase/migrations/           # Apply the single clean v1 schema
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
- **Local storage is per-browser, not per-person.** The device id is a scoping key, not an
  identity. On a shared or public deployment, configure Supabase so sessions are tied to a
  verified account.
- **Local storage on Vercel is temporary.** Only `/tmp` is writable there and it is evicted
  freely. Set `TRUTHLENS_DATA_DIR` to a real volume, or use Supabase.
- **Benchmark reports behaviour, not accuracy** — there is no labelled ground truth, and calling
  decisiveness "accuracy" would be a fabrication.

---

## License & attribution

Inspired by the **Real-LOD (ICLR 2025)** research workflow for agentic vision-language grounding
and verification.
