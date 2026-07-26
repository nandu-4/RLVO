# RLVO — Verification-First Proctoring & Agentic Vision-Language Re-alignment

**One thesis: raw computer-vision output cannot be trusted — in captions or in proctoring — until an agent verifies it against the pixels.**

RLVO applies that thesis twice in one codebase:

1. **Verification-first exam proctoring** — a **two-stage** system where cheap real-time detectors (MediaPipe geometry, COCO-SSD) *propose* flags, and an **agentic VLM verifier** *disposes*: every high-severity flag is fact-checked against the captured frame before any trust penalty applies. Refuted flags are dismissed with reasoning; every mark against a candidate carries a visual evidence trail they could appeal. Continuous video never leaves the browser — only single flagged frames are verified, and that layer is an explicit opt-in toggle.
2. **Vision-language re-alignment** — demonstrates how VLM captions hallucinate (invented brands, backstory, emotions) and fixes them with the same verify-then-rewrite agentic loop, claim by claim.

This attacks the two documented failures of commercial proctoring (Proctorio, ExamSoft, ProctorU): black-box false positives that punish innocent behavior, and full video streams leaving the candidate's machine.

Inspired by the Real-LOD research workflow (agentic refinement of noisy language descriptions for open-vocabulary detection), re-imagined as an interactive web product.

---

## Table of Contents

- [What We Built and Why](#what-we-built-and-why)
- [Did We Train Any Models?](#did-we-train-any-models)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Pipeline 1 — Image Caption Re-alignment](#pipeline-1--image-caption-re-alignment)
- [Pipeline 2 — Video Understanding](#pipeline-2--video-understanding)
- [Pipeline 3 — Real-time Proctoring](#pipeline-3--real-time-proctoring)
- [Python Reference Implementation](#python-reference-implementation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Supabase Edge Functions](#supabase-edge-functions)
- [Evaluation — How We Know It Works](#evaluation--how-we-know-it-works)
- [Export Reports](#export-reports)
- [Interview Talking Points](#interview-talking-points)

---

## What We Built and Why

Vision-language models (VLMs) are fluent but **overconfident**: asked to describe an image, they invent brands, materials, backstory, and emotions that are not visible. This is the *hallucination problem*, and it makes raw VLM output unusable for downstream tasks (accessibility, dataset labeling, open-vocabulary detection).

RLVO demonstrates the problem and the fix side by side:

- **Stage 1 (`generate-caption`)** deliberately produces a hallucination-rich caption — high temperature (1.3), a prompt that *demands* brands, backstory and confident assertions. This simulates the noisy language descriptions that the Real-LOD paper starts from.
- **Stage 2 (`refine-caption`)** runs the **agentic re-alignment loop**: decompose the caption into atomic claims, verify each claim against the image with a deterministic fact-checker pass, then rewrite the caption using only verified content. The UI shows the real per-claim evidence log (✓ CORRECT / ✗ WRONG / ~ UNCERTAIN → correction).

The proctoring dashboard applies the same "trust through verification" idea to live video: every detection channel is measured against a **per-user calibrated baseline** instead of hard-coded absolutes, which eliminates false positives from camera angle and seating position.

---

## Did We Train Any Models?

**No model was trained or fine-tuned — and that is a deliberate engineering decision worth defending in an interview.** The system composes three pretrained models and gets its accuracy from *calibration, prompt engineering, and an agentic verification architecture* instead of gradient updates:

| Model | Type | Where it runs | What we did instead of training |
|---|---|---|---|
| **Google Gemini 2.5 Flash** | Vision-language model | Cloud (via Lovable AI Gateway / Google API) | Prompt engineering: adversarial "hallucinate confidently" prompt for stage 1; strict JSON fact-checker + grounded-rewrite prompts at temperature 0–0.2 for stage 2 |
| **MediaPipe Face Mesh** | 468-landmark face model (Google, pretrained) | Browser (WASM, ~30 fps) | Built geometric detectors on top of the landmarks (yaw ratio, face aspect ratio, iris offset) and calibrated per-session baselines |
| **COCO-SSD (lite_mobilenet_v2)** | Object detector, 80 COCO classes (pretrained) | Browser (TensorFlow.js, WebGL) | Tuned inference pipeline: downscaled input, confidence threshold + multi-hit confirmation, decoupled detection timer |

Why not fine-tune? (a) No labeled training data exists for "this specific user's webcam at this angle" — per-session calibration solves what fine-tuning would; (b) hallucination is better fixed at the *system* level (verify-then-rewrite) than by fine-tuning a captioner, and the loop transfers to any VLM; (c) browser-side models must stay small — swapping in a custom-trained detector would cost 10–100× the size for marginal gain on the single "cell phone" class we care about.

What *was* tuned, empirically, against real session reports:

- Head-turn yaw threshold `0.33` (raised from 0.25 to cut borderline triggers)
- Gaze deviation `5%` of eye width with a **leaky counter** (previously 7% + hard reset — which caused sessions to log 0 gaze events; a single frame of iris jitter kept wiping the counter)
- Phone confidence `0.45` + 2-hit confirmation (previously a single `0.6` gate — the lite model rarely scores phones that high, so detection felt slow/missing)
- Look-down: 10% face-compression sustained 20 frames; calibration window 90 frames

---

## Architecture

```
┌────────────────────────── Browser (React + TS) ──────────────────────────┐
│                                                                          │
│  ImageRefinement.tsx      VideoRefinement.tsx      Proctoring.tsx        │
│        │                        │                       │                │
│        │ base64 image           │ canvas-extracted      │ webcam stream  │
│        │                        │ frames (base64 JPEG)  ▼                │
│        │                        │              useProctoring.ts          │
│        │                        │              ├─ MediaPipe Face Mesh    │
│        │                        │              │  (rAF loop, ~30 fps)    │
│        │                        │              ├─ COCO-SSD (700 ms timer,│
│        │                        │              │  320px canvas, WebGL)   │
│        │                        │              └─ visibility/blur events │
│        ▼                        ▼                                        │
│   invokeAi() ──── VITE_BACKEND switch ────────────────────┐              │
└───────────┼───────────────────────────────────────────────┼──────────────┘
            ▼ supabase (default)                            ▼ python (local)
┌─── Supabase Edge Functions (Deno) ───┐      ┌─── FastAPI server.py ──────┐
│  generate-caption   refine-caption   │      │  image_refinement.py       │
│  analyze-video                       │      │  video_refinement.py       │
└──────────────┬───────────────────────┘      └──────────────┬─────────────┘
               ▼                                             ▼
     Lovable AI Gateway ──► Google Gemini 2.5 Flash ◄── Google AI API
```

Key properties:

- **Proctoring is 100% client-side** — video frames never leave the browser; only the alert log exists as data. Privacy by architecture, not policy.
- **Dual backend** — the same UI can hit Supabase Edge Functions (production) or a local Python FastAPI server (development / offline demo) via one env var.
- **Models load from CDN at session start** — no bundle bloat; the app ships ~0 MB of ML weights.

---

## Tech Stack

**Frontend**
- [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/) (SWC)
- [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/) (Radix primitives)
- [React Router DOM v6](https://reactrouter.com/), [TanStack React Query v5](https://tanstack.com/query), [Sonner](https://sonner.emilkowal.ski/)

**AI / ML (client-side)**
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) — 468 facial landmarks + iris refinement (WASM)
- [TensorFlow.js](https://www.tensorflow.org/js) + [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) `lite_mobilenet_v2` — phone detection (WebGL backend)

**Backend**
- [Supabase Edge Functions](https://supabase.com/) (Deno) + Lovable AI Gateway → **Google Gemini 2.5 Flash**
- Alternative: local **Python FastAPI** server calling the Google Generative Language API directly

---

## Pipeline 1 — Image Caption Re-alignment

**Route:** `/image-refinement`

1. **Upload** any image (PNG/JPG/WEBP) — converted to a base64 data URL in the browser.
2. **`generate-caption`** — Gemini at temperature 1.3 with an adversarial prompt that *forces* hallucination: specific brands, backstory, emotions, zero hedging. This is the "before" exhibit.
3. **`refine-caption`** — the agentic loop, two deterministic passes:
   - **Pass 1 — VERIFY** (temperature 0): decompose the raw caption into atomic claims; classify each by aspect (`category / attribute / accessory / relation / location / behavior`); output a JSON verdict per claim — `CORRECT`, `WRONG` (includes anything not visible: brands, backstory, emotion), or `UNCERTAIN` with a grounded correction.
   - **Pass 2 — REWRITE** (temperature 0.2): compose the final 3–5 sentence caption using **only** CORRECT claims and the corrections of UNCERTAIN ones. WRONG claims are dropped.
4. The UI streams the **real evidence log** — every claim with its verdict and correction — then shows the refined caption beside the original.

If Pass 1 returns unparseable JSON the function degrades gracefully to a single-pass grounded rewrite, so the demo never hard-fails.

---

## Pipeline 2 — Video Understanding

**Route:** `/video-refinement`

| Mode | Frames | Output |
|---|---|---|
| **Video Summary** | 6 evenly-spaced | One 2–3 sentence grounded summary of the whole clip |
| **Time Capsule** | 8 evenly-spaced | One caption per frame with timestamp, rendered as a timeline |

- Frame extraction is pure browser: seek the `<video>` element, draw to `<canvas>`, export base64 JPEG. No ffmpeg, no upload of the raw video.
- Time Capsule captions all frames **in parallel** (`Promise.all`) — latency is ~1 request instead of ~8 sequential ones.
- Both prompts carry explicit grounding rules: no invented names/brands/sounds/dialogue, no events "between" frames, no hedging words; temperature 0.2.

---

## Pipeline 3 — Real-time Proctoring

**Route:** `/proctoring`

Two pretrained models run simultaneously in the browser, plus DOM events:

### Baseline Calibration (first 3 seconds)

The first **90 frames** record the user's neutral face aspect ratio and iris offset. Every later threshold is a **delta from this baseline** — so camera angle, distance, and seating position cannot cause false positives.

### Head Turn (Yaw) — MediaPipe, every frame

```
yaw = (noseTip.x − earMidX) / (earSpan / 2)      alert if |yaw| > 0.33
```

### Gaze Tracking (Iris) — MediaPipe, every frame

Iris landmarks 468/473 (needs `refineLandmarks: true`), normalized by eye width relative to the eye-corner midpoint:

- Deviation > **5% of eye width** from calibrated baseline
- **Leaky counter**: deviated frames increment a score; centered frames *drain* it by 2 instead of resetting — so one frame of iris jitter can't erase a sustained glance. Alert at score 10 (~0.4 s).
- Skipped while the head itself is turned (the head-turn alert covers that; iris geometry is unreliable at high yaw). Gaze therefore specifically catches **eyes-only glancing**.

### Looking Down / Phone Use (faceAR) — MediaPipe, every frame

```
faceAR = faceHeight / faceWidth        (foreshortens when head tilts down)
```

A **10% compression** below baseline sustained **20 frames** (~0.7 s) → `looking_down` alert. This catches phone-in-lap use that the object detector cannot see.

### Phone in Frame (COCO-SSD) — independent 700 ms timer

- Runs on its **own timer**, decoupled from the Face Mesh loop, so face-tracking latency never delays it
- Inference on a **320px downscaled canvas** (not the full 1280×720 frame) → far faster per-detect on the WebGL backend, which is explicitly selected and **warmed up** at session start (first inference compiles shaders)
- `cell phone` class at confidence ≥ **0.45**, confirmed across **2 consecutive checks** before alerting — catches phones much sooner than a single high-confidence gate while keeping false positives out

### Tab / Focus — DOM events

`visibilitychange` (tab switch / minimize) and window `blur` (focus to another app).

### Stage 2 — Agentic Flag Verification (the novel layer)

The detectors above are fast but dumb geometry — every proctoring product has them, and their false positives are where real students get hurt. RLVO adds what none of them have: **an adversarial VLM verifier that fact-checks each high-severity flag before it counts.**

Flow, implemented in `useProctoring.ts` + `supabase/functions/verify-flag`:

1. A verifiable flag fires (`phone_detected`, `multiple_faces`, `no_face`, `looking_down` — visual claims; tab switches aren't visual, head turns are too frequent to verify economically).
2. The current frame is captured (640px JPEG) as the **evidence exhibit** — the trust penalty is **deferred**.
3. The frame + the detector's claim go to the verifier, prompted as an *adversarial skeptic* ("detectors are frequently wrong; a wrong CONFIRMED unfairly accuses a real person; confirm only what you clearly see") with per-flag-type questions that encode known false-positive modes (a face in a poster is not a second person; a remote control is not a phone; glancing at the keyboard is not phone use). Temperature 0, strict JSON verdict.
4. The verdict disposes:
   - **CONFIRMED** → full trust penalty, evidence reasoning attached
   - **REFUTED** → **dismissed**: no penalty, struck through in the log, counted in "Dismissed by AI"
   - **UNCERTAIN** → half penalty
   - Verifier unreachable → full penalty, marked `unverified` — so blocking the network can never be used to dodge penalties (fail-safe, not fail-open)
5. The alert log shows the verdict badge, the verifier's reasoning, and the flagged-frame thumbnail. Exports carry the full audit trail.

Privacy trade-off, stated honestly: continuous video never leaves the browser; **single flagged frames** are sent for verification only when the "Agentic flag verification" toggle is on (visible consent control on the dashboard; off = fully offline monitoring with classic immediate penalties).

### Trust Score

Starts at 100. Unverified channels decay immediately per severity: high −6, medium −3, low −1, info 0. Verified channels decay only on the verifier's verdict: confirmed −6, uncertain −3, dismissed 0.

---

## Python Reference Implementation

`python/` contains standalone mirrors of all three pipelines — used for offline runs, benchmarking, and to make the system explainable in academic settings (IOMP report, viva):

| Script | Mirrors | Notes |
|---|---|---|
| `image_refinement.py` | `generate-caption` + `refine-caption` | Same two-pass verify→rewrite loop, Gemini direct API, retry with exponential backoff |
| `video_refinement.py` | `analyze-video` | OpenCV frame extraction, Summary + Time Capsule modes |
| `proctoring.py` | `useProctoring.ts` | MediaPipe Face Mesh + YOLOv8n phone detection (desktop-grade equivalent of COCO-SSD) |
| `server.py` | Supabase functions layer | FastAPI server exposing the same routes, so the React app can run fully local with `VITE_BACKEND=python` |

```bash
pip install -r python/requirements.txt        # Python 3.9–3.11 (mediapipe constraint)
export GEMINI_API_KEY=...                     # free at aistudio.google.com/apikey
python python/image_refinement.py photo.jpg
```

---

## Project Structure

```
src/
├── components/
│   ├── Navigation.tsx          # Top nav bar linking all 3 pages
│   └── ui/                     # shadcn/ui component library
├── hooks/
│   └── useProctoring.ts        # All proctoring logic — MediaPipe, COCO-SSD, alerts, export
├── integrations/
│   ├── aiClient.ts             # invokeAi() — VITE_BACKEND switch (supabase | python)
│   └── supabase/client.ts      # Supabase client (reads VITE_ env vars)
├── pages/
│   ├── Index.tsx               # Landing page
│   ├── ImageRefinement.tsx     # Caption re-alignment demo
│   ├── VideoRefinement.tsx     # Video summary + time capsule
│   └── Proctoring.tsx          # Proctoring dashboard
└── App.tsx                     # Routes

supabase/functions/
├── generate-caption/index.ts   # Stage 1 — hallucination-rich raw caption
├── refine-caption/index.ts     # Stage 2 — two-pass agentic re-alignment
└── analyze-video/index.ts      # Video summary + parallel frame captions

python/                         # Reference implementation (see above)
```

---

## Getting Started

```bash
npm install
npm run dev          # → http://localhost:8080
npm run build        # production build
```

Prerequisites: Node 18+; for the cloud backend, a Supabase project with the three Edge Functions deployed and `LOVABLE_API_KEY` set; or run the Python backend locally instead.

---

## Environment Variables

`.env` in the project root:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key

# Optional — switch the AI backend (default: supabase)
VITE_BACKEND=python
VITE_PYTHON_API=http://localhost:8000
```

Supabase Edge Function secret (Dashboard → Settings → Edge Functions → Secrets):

```
LOVABLE_API_KEY=your-lovable-api-key
```

Python backend: `GEMINI_API_KEY` in the shell environment.

> `.env` is gitignored — secrets never enter version control.

---

## Supabase Edge Functions

```bash
supabase functions deploy generate-caption
supabase functions deploy refine-caption
supabase functions deploy analyze-video
supabase functions deploy verify-flag
```

| Function | Input | Output |
|---|---|---|
| `generate-caption` | `{ image }` (base64 data URL) | `{ caption }` |
| `refine-caption` | `{ image, rawCaption }` | `{ refinedCaption, logs[], verdicts[], stats }` |
| `analyze-video` | `{ frames[], mode: "summary" \| "timecapsule" }` | `{ summary }` or `{ captions[] }` |
| `verify-flag` | `{ frame, flagType, claim }` | `{ verdict: "CONFIRMED" \| "REFUTED" \| "UNCERTAIN", evidence, confidence }` |

All functions send CORS headers and return HTTP 500 with `{ error }` on failure. `refine-caption` degrades to single-pass if the verification JSON fails to parse.

---

## Evaluation — How We Know It Works

**Caption re-alignment** — claim-level accounting, produced by the system itself:
- Every refinement run returns `verdicts[]` and `stats { correct, wrong, uncertain }` — i.e., how many claims the raw caption got wrong and what survived verification. Across typical photos, the adversarial stage-1 caption produces a majority of WRONG/UNCERTAIN claims (invented brands, backstory, emotion), and the refined caption retains only visually grounded content — the per-claim log is the audit trail.
- This mirrors hallucination metrics from the literature (CHAIR-style object hallucination counting), done claim-by-claim rather than object-by-object.

**Flag verification** — measurable false-positive reduction:
- Every session report now separates *detector proposals* from *verified violations*: the "Dismissed by AI" count is literally the number of false accusations the system prevented. Run a session, hold up a TV remote (classic phone false-positive), and watch the verifier refute it with written reasoning — that demo *is* the evaluation.

**Proctoring** — session reports as ground truth:
- Every session exports CSV/JSON with per-channel counts. Detection thresholds were tuned against real recorded sessions: e.g., a session that visibly included off-screen glances but logged `Gaze Aways: 0` exposed the hard-reset counter bug; a phone held in frame for seconds before alerting exposed the full-resolution + 0.6-threshold bottleneck. Both fixes are documented in [Did We Train Any Models?](#did-we-train-any-models).
- Latency: phone checks run every 700 ms on a 320px canvas with a warmed WebGL backend — worst-case time-to-alert ≈ 700 ms × 2 confirmations + inference ≈ **~1.5–2 s**, versus 5–15+ s before.

**Engineering checks** — `tsc --noEmit` clean, production `vite build` clean, Python modules compile; manual end-to-end runs of all three pages.

---

## Export Reports

At session end, **Export CSV** or **Export JSON** downloads the full report: every alert (timestamp, type, message, severity) plus a summary — trust score, duration, and per-channel counts including phone detections.

```json
{
  "exportedAt": "2026-07-26T10:30:00.000Z",
  "durationSeconds": 900,
  "trustScore": 74,
  "stats": { "headTurns": 2, "gazeAways": 1, "noFaceEvents": 0,
             "multipleFaceEvents": 0, "lookingDownEvents": 1,
             "phoneDetectedEvents": 2, "tabSwitches": 0 },
  "alerts": [ { "type": "head_turn_right", "message": "Head turned right (41% deviation)", "severity": "medium", "time": "10:16:34 AM", "timestamp": 1747131394000 } ]
}
```

---

## Interview Talking Points

- **What's genuinely new:** commercial proctoring stops at stage 1 (raw detectors → raw accusations). RLVO adds stage 2: an adversarial VLM verifier that fact-checks every high-severity flag against the frame before it penalizes, dismisses false positives with written reasoning, and attaches the frame as an appealable evidence exhibit. "Detectors propose, the verifier disposes" — one sentence nobody will confuse with Proctorio.
- **Fail-safe design:** if the verifier is unreachable, the standard penalty applies — so a candidate can't dodge penalties by blocking the network. Verification can only ever *help* an honest candidate, never a dishonest one.
- **The privacy trade-off, owned explicitly:** continuous video never leaves the browser; single flagged frames are sent only under a visible consent toggle. Being able to articulate why this is still a radically smaller footprint than streaming everything is itself a talking point.
- **Problem framing:** VLM hallucination is a *system* problem, not just a model problem — RLVO fixes it with verify-then-rewrite architecture rather than fine-tuning, so the fix is model-agnostic.
- **Agentic loop:** decompose → verify each claim at temperature 0 → rewrite from verdicts → emit the evidence log. Two LLM calls, deterministic where it matters, graceful degradation when parsing fails.
- **Why no training:** per-session calibration replaces personalization-by-fine-tuning; pretrained detectors are sufficient when the *pipeline* around them is tuned (downscaling, thresholds, confirmation logic, backend selection).
- **Privacy:** proctoring video never leaves the browser — the models come to the data, not the reverse.
- **Real debugging story:** the gaze channel logged zero events in production sessions. Root cause: a hard-reset consecutive-frame counter destroyed by single-frame iris jitter. Fix: leaky counter + lower threshold + gating gaze behind head pose. The phone channel was slow because inference ran at 1280×720 on a possibly-CPU TF.js backend inside the face-tracking loop at a 0.6 confidence gate; fix: dedicated timer, 320px canvas, explicit WebGL + warm-up, 0.45 with 2-hit confirmation.
- **Trade-offs made:** two-pass refinement doubles latency and cost for accuracy and auditability; parallel frame captioning trades burst rate-limit risk for 8× latency win; leaky counter trades a slightly slower alert for drastically fewer missed detections.

---

## License

MIT
