# RLVO - Python Reference Implementation

Python equivalents of the three RLVO pipelines. The main app is TypeScript/React,
but these scripts mirror the same algorithms for offline use, benchmarking, and
to make the system explainable in academic settings (IOMP report, viva, etc.).

| Script | Mirrors | Purpose |
|---|---|---|
| `image_refinement.py` | `generate-caption` + `refine-caption` edge functions | VLM caption + agentic re-alignment loop |
| `video_refinement.py` | `analyze-video` edge function | Frame extraction + Summary or Time Capsule mode |
| `proctoring.py` | `useProctoring.ts` hook | MediaPipe Face Mesh + YOLOv8 phone detection |

---

## Install

```bash
pip install -r requirements.txt
```

> `mediapipe` requires Python 3.9-3.11 on Windows. `ultralytics` will auto-download
> the `yolov8n.pt` weights (~6 MB) on first run.

---

## Set the API key

The image and video scripts call the **Google Gemini API** directly (free key at https://aistudio.google.com/apikey).
Set the key in your shell first:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key-here"
```

```bash
# Linux/Mac
export GEMINI_API_KEY=your-key-here
```

---

## Run

### 1. Image refinement

```bash
python image_refinement.py path/to/image.jpg
```

Output:
```
[1/3] Encoding image: path/to/image.jpg
[2/3] Generating raw caption...
   Raw caption     : A person is sitting at a desk with a laptop.
[3/3] Running agentic re-alignment...
   Refined caption : A woman in a blue shirt is typing on a silver MacBook ...
```

### 2. Video refinement

```bash
# Single 2-3 sentence summary
python video_refinement.py path/to/video.mp4 summary

# One caption per frame (timeline)
python video_refinement.py path/to/video.mp4 timecapsule
```

### 3. Proctoring

```bash
# Webcam mode (default)
python proctoring.py

# Offline mode on a recorded video
python proctoring.py session.mp4
```

Press **q** to quit. Exits with a full `proctoring_report.json` containing trust
score, per-violation stats, and the alert timeline - same schema as the browser
export.

---

## Algorithm parity

All thresholds match the browser implementation exactly:

| Constant | Value | Meaning |
|---|---|---|
| `CALIB_FRAMES` | 90 (~3 s) | Baseline calibration window |
| `YAW_THRESHOLD` | 0.33 | Head-turn fires above this fraction |
| `LOOK_DOWN_AR_THRESHOLD` | 0.10 (10%) | Face Aspect Ratio drop threshold |
| `LOOK_DOWN_SUSTAINED` | 20 frames | ~0.7 s before firing alert |
| `GAZE_DELTA` | 0.07 (7%) | Iris-offset deviation threshold |
| `GAZE_SUSTAINED` | 10 frames | Frames of held gaze before firing |
| `PHONE_CONF` | 0.6 | YOLO "cell phone" min confidence |
| `PHONE_EVERY_N` | 10 | Run YOLO every 10th frame |

Severity decay (trust score impact):

| Severity | Decay |
|---|---|
| high | -6 |
| medium | -3 |
| low | -1 |
| info | 0 |

---

## Why Python at all?

The production system runs entirely in the browser (no Python backend needed).
This folder exists for three reasons:

1. **Reproducible offline benchmarking** of caption refinement quality.
2. **Academic deliverables** - the IOMP report can cite the same algorithms in
   a language graders are more familiar with.
3. **Easier debugging** of the proctoring detection math without launching a
   webcam in the browser.

The browser version uses **COCO-SSD** (TensorFlow.js); the Python version uses
**YOLOv8n** (Ultralytics). Both detect the COCO `cell phone` class; YOLOv8 is
chosen here because it has cleaner Python APIs and ships pre-trained weights.
