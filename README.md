# RLVO — Agentic Re-alignment for Vision-Language

RLVO is an AI-powered web application that combines vision-language alignment (reducing hallucinations in AI-generated captions) with real-time exam proctoring. It runs entirely in the browser using React, TypeScript, and Supabase Edge Functions.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Pages](#pages)
  - [Image Refinement](#image-refinement)
  - [Video Refinement](#video-refinement)
  - [Proctoring Dashboard](#proctoring-dashboard)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Supabase Edge Functions](#supabase-edge-functions)
- [Proctoring — How Detection Works](#proctoring--how-detection-works)
- [Export Reports](#export-reports)

---

## Features

| Feature | Description |
|---|---|
| Image Caption Refinement | Upload an image, get an AI-generated caption, then run multi-cycle agentic re-alignment to improve accuracy |
| Video Understanding | Upload a video, extract key frames automatically, and generate either a unified summary or a per-frame Time Capsule timeline |
| Real-time Proctoring | Webcam-based AI monitoring for online exams — head turns, gaze tracking, phone detection, tab switches, multiple faces |
| Trust Score | Live score that decays on each violation; exportable at session end |
| Export | Download full session reports as CSV or JSON |

---

## Tech Stack

**Frontend**
- [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/) (build tool)
- [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/) (component library)
- [React Router DOM v6](https://reactrouter.com/) (routing)
- [TanStack React Query v5](https://tanstack.com/query) (server state)
- [Sonner](https://sonner.emilkowal.ski/) (toast notifications)

**AI / ML (client-side, no server needed)**
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) — 468 facial landmarks + iris tracking, loaded from CDN
- [TensorFlow.js](https://www.tensorflow.org/js) + [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) — real-time object detection for phone detection, loaded from CDN

**Backend**
- [Supabase](https://supabase.com/) — Edge Functions (Deno runtime), environment secrets
- [Lovable AI Gateway](https://lovable.dev/) — proxied access to Google Gemini 2.5 Flash for image and video captioning

---

## Pages

### Image Refinement

**Route:** `/image-refinement`

Upload any image (PNG, JPG, WEBP) and the system:

1. Sends the image to the `generate-caption` Supabase function, which calls a Vision-Language Model to produce a raw caption.
2. Sends the image + raw caption to the `refine-caption` function, which runs an agentic re-alignment loop — iteratively scoring and improving the caption using a reflection model.
3. Displays the refined caption alongside the original, with real-time processing logs.

**Key stat:** ~60% improvement in alignment accuracy vs. single-pass captioning.

---

### Video Refinement

**Route:** `/video-refinement`

Upload a video (MP4, WebM, MOV) and choose one of two analysis modes:

| Mode | Frames extracted | Output |
|---|---|---|
| **Video Summary** | 6 evenly-spaced frames | A single 2–3 sentence summary of the whole video |
| **Time Capsule** | 8 evenly-spaced frames | One caption per frame with timestamp, shown as a visual timeline |

Frame extraction happens entirely in the browser using HTML5 `<canvas>`. Frames are sent as base64 JPEG images to the `analyze-video` Supabase function which calls Gemini 2.5 Flash.

---

### Proctoring Dashboard

**Route:** `/proctoring`

A full real-time AI proctoring system powered by two models running simultaneously in the browser:

**MediaPipe Face Mesh** (every frame, ~30 fps)
- Head yaw — detects left/right head turns using ear-span landmark geometry
- Iris gaze tracking — 10 iris landmarks measure lateral gaze offset from a calibrated baseline
- Face Aspect Ratio (faceAR) compression — detects downward head tilt toward a phone via facial foreshortening
- Multiple face detection — alerts when more than one person appears in frame

**COCO-SSD / TensorFlow.js** (every 10th frame)
- Object detection across 80 COCO classes
- Fires a `phone_detected` alert when a cell phone is visible with confidence > 60%

**Tab Switch Detection**
- `visibilitychange` event — catches browser tab switches and window minimise
- `blur` event — catches focus loss to another application

**Live Detection Status Panel** — 6 detection channels updated every 200 ms, displayed without any canvas overlay on the video feed.

---

## Project Structure

```
src/
├── components/
│   ├── Navigation.tsx          # Top nav bar linking all 3 pages
│   └── ui/                     # shadcn/ui component library
├── hooks/
│   ├── useProctoring.ts        # All proctoring logic — MediaPipe, COCO-SSD, alerts, export
│   └── use-toast.ts            # Toast hook (shadcn)
├── integrations/
│   └── supabase/
│       ├── client.ts           # Supabase client (reads VITE_ env vars)
│       └── types.ts            # Auto-generated database types
├── pages/
│   ├── Index.tsx               # Landing / home page
│   ├── ImageRefinement.tsx     # Image caption refinement
│   ├── VideoRefinement.tsx     # Video understanding (summary + time capsule)
│   ├── Proctoring.tsx          # Proctoring dashboard UI
│   └── NotFound.tsx            # 404 fallback
└── App.tsx                     # Route definitions

supabase/
└── functions/
    ├── generate-caption/       # VLM caption generation for images
    │   └── index.ts
    ├── refine-caption/         # Agentic caption re-alignment
    │   └── index.ts
    └── analyze-video/          # Video frame analysis (summary + time capsule)
        └── index.ts
```

---

## Getting Started

### Prerequisites

- Node.js 18+ (or Bun)
- A Supabase project with the three Edge Functions deployed
- `LOVABLE_API_KEY` secret configured in the Supabase dashboard

### Install dependencies

```bash
npm install
```

### Start the development server

```bash
npm run dev
```

The app runs at `http://localhost:8080`.

### Build for production

```bash
npm run build
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key
```

The Edge Functions require one secret set inside the Supabase dashboard (Settings → Edge Functions → Secrets):

```
LOVABLE_API_KEY=your-lovable-api-key
```

---

## Supabase Edge Functions

Deploy all three functions with the Supabase CLI:

```bash
supabase functions deploy generate-caption
supabase functions deploy refine-caption
supabase functions deploy analyze-video
```

| Function | Input | Output |
|---|---|---|
| `generate-caption` | `{ image: string }` (base64 data URL) | `{ caption: string }` |
| `refine-caption` | `{ image: string, rawCaption: string }` | `{ refinedCaption: string, logs: string[] }` |
| `analyze-video` | `{ frames: string[], mode: "summary" \| "timecapsule" }` | `{ summary: string }` or `{ captions: string[] }` |

All functions include CORS headers for browser access and return HTTP 500 with `{ error }` on failure.

---

## Proctoring — How Detection Works

### Head Turn (Yaw)

```
yaw = (noseTip.x − earMidX) / (earSpan / 2)
```

Alert fires when `|yaw| > 0.33`. Cooldown: 2.5 s per direction.

### Gaze Tracking (Iris)

Iris landmarks 468 (left eye) and 473 (right eye) require `refineLandmarks: true`. Each iris position is normalised by eye width relative to the eye-corner midpoint. The delta from the calibrated baseline must exceed **7% of eye width** sustained for **10 consecutive frames** before an alert fires.

### Looking Down / Phone Use (faceAR)

```
faceAR       = faceH / faceWidth
faceARDelta  = (faceAR − baseline) / baseline
```

When the head tilts forward toward a phone, the face foreshortens vertically — `faceAR` drops. A **10% drop sustained for 20 frames** (~0.7 s at 30 fps) fires a `looking_down` alert.

### Phone in Frame (COCO-SSD)

Every 10th video frame is passed to the TensorFlow.js COCO-SSD lite model. If any prediction has `class === "cell phone"` with `score > 0.6`, a `phone_detected` alert fires. 5 s cooldown prevents spam.

### Baseline Calibration

The first 90 frames (~3 seconds) after session start are used to record neutral values for `faceAR` and iris gaze offset. All subsequent thresholds are measured as deltas from this baseline, eliminating false positives from camera angle and seating position.

### Trust Score Decay

| Severity | Trust decay |
|---|---|
| High | −6 |
| Medium | −3 |
| Low | −1 |
| Info | 0 |

---

## Export Reports

At session end, click **Export CSV** or **Export JSON** to download the full report.

**CSV** — every alert row (timestamp, type, message, severity) plus a summary section.

**JSON structure:**
```json
{
  "exportedAt": "2025-05-13T10:30:00.000Z",
  "sessionStart": "2025-05-13T10:15:00.000Z",
  "durationSeconds": 900,
  "trustScore": 74,
  "stats": {
    "headTurns": 2,
    "gazeAways": 1,
    "noFaceEvents": 0,
    "multipleFaceEvents": 0,
    "lookingDownEvents": 1,
    "phoneDetectedEvents": 2,
    "tabSwitches": 0
  },
  "alerts": [
    {
      "id": "abc123",
      "time": "10:16:34 AM",
      "timestamp": 1747131394000,
      "type": "head_turn_right",
      "message": "Head turned right (41% deviation)",
      "severity": "medium"
    }
  ]
}
```

---

## License

MIT
