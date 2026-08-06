"""
RLVO - Video Refinement (Python, Google Gemini direct)
------------------------------------------------------
Mirrors the analyze-video edge function, but calls Gemini's REST API directly.

Modes:
    summary     : 6 evenly-spaced frames -> single 2-3 sentence summary
    timecapsule : 8 evenly-spaced frames -> one caption per frame

Env:
    GEMINI_API_KEY  -> Get one free at https://aistudio.google.com/apikey

CLI:
    python video_refinement.py path/to/video.mp4 summary
    python video_refinement.py path/to/video.mp4 timecapsule
"""

import os
import sys
import re
import time
import base64
import requests
# cv2 is only needed for offline frame extraction; lazy-imported in extract_frames
# so the FastAPI server can import this module without requiring opencv-python.

MODEL = "gemini-2.5-flash-lite"
API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# Frame counts are no longer fixed here. They come from shared/video-sampling.json via
# video_sampling.frame_count_for(), which the browser reads too — the literals 6 and 8 used to
# live in both places and drift independently.
from video_sampling import frame_count_for, frame_indices_for


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get one free at https://aistudio.google.com/apikey "
            "and add it to .env or your shell environment."
        )
    return key


def _split_data_url(data_url: str) -> tuple[str, str]:
    m = re.match(r"data:([^;]+);base64,(.+)", data_url, flags=re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return "image/jpeg", data_url


RETRYABLE = {429, 500, 502, 503, 504}
MAX_RETRIES = 4


def _call_gemini(parts: list, max_tokens: int = 200, temperature: float = 0.4) -> str:
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_err = ""
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            API_BASE,
            params={"key": _api_key()},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        if resp.ok:
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {data}")
            parts_out = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts_out).strip()
            if not text:
                raise RuntimeError(f"Gemini returned empty text: {data}")
            return text

        last_err = f"Gemini API {resp.status_code}: {resp.text}"
        if resp.status_code not in RETRYABLE:
            break
        wait = 2 ** attempt
        print(f"[gemini retry] {resp.status_code} - sleeping {wait}s "
              f"(attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
        time.sleep(wait)

    raise RuntimeError(last_err)


# ---------------------------------------------------------------------------
# Frame extraction (offline CLI use only)
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, n_frames: int) -> list:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Video reports zero frames")

    # Indices span the whole video, first frame to last. The previous `i * (total // n_frames)`
    # stepping stopped short of the end, so the closing portion was never sampled.
    frames = []
    for idx in frame_indices_for(total, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{b64}")

    cap.release()
    if not frames:
        raise RuntimeError("Failed to extract any frames")
    return frames


# ---------------------------------------------------------------------------
# Mode 1: Video Summary
# ---------------------------------------------------------------------------
def summarize_video(frames: list) -> str:
    parts: list = [{
        "text": (
            "These are key frames from a video shown in chronological order. "
            "Analyze them and provide a concise summary (2-3 sentences) "
            "describing the main events, actions, and subjects in the video. "
            "Focus on what actually happens, avoiding hallucinations."
        )
    }]
    for frame in frames:
        mime, b64 = _split_data_url(frame)
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    return _call_gemini(parts, max_tokens=200)


# ---------------------------------------------------------------------------
# Mode 2: Time Capsule
# Batched: sends all N frames in ONE Gemini call to stay under free-tier quota.
# ---------------------------------------------------------------------------
def caption_per_frame(frames: list) -> list:
    n = len(frames)
    parts: list = [{
        "text": (
            f"You are looking at {n} key frames extracted from a video, shown in "
            "chronological order. Write exactly one concise caption for each frame "
            "(one sentence, focused on the main subject, action, and visible "
            "elements; be accurate and avoid hallucinations).\n\n"
            f"Return your answer as exactly {n} lines, each starting with the "
            "frame number followed by a period and a space, like:\n"
            "1. caption for frame one.\n"
            "2. caption for frame two.\n"
            "...and so on. Do not add any other text, headers, or explanation."
        )
    }]
    for frame in frames:
        mime, b64 = _split_data_url(frame)
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})

    raw = _call_gemini(parts, max_tokens=100 * n + 200, temperature=0.4)
    return _parse_numbered_captions(raw, n)


def _parse_numbered_captions(text: str, expected: int) -> list:
    """Parse '1. ...\\n2. ...' lines into a list of length expected."""
    import re as _re
    captions: list[str] = []
    for line in text.splitlines():
        m = _re.match(r"\s*(\d+)\s*[\.\):-]\s*(.+)", line)
        if m:
            captions.append(m.group(2).strip())
    if len(captions) < expected:
        # Fallback: pad with the original raw text so we don't crash
        while len(captions) < expected:
            captions.append("(no caption returned)")
    return captions[:expected]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_pipeline(video_path: str, mode: str) -> None:
    if mode not in ("summary", "timecapsule"):
        raise ValueError('mode must be "summary" or "timecapsule"')

    # Duration drives the count. CAP_PROP_FRAME_COUNT / FPS is the reliable way to get it
    # without decoding the file twice.
    import cv2
    probe = cv2.VideoCapture(video_path)
    fps = probe.get(cv2.CAP_PROP_FPS) or 0
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()
    duration = (total / fps) if fps > 0 and total > 0 else 0.0
    n = frame_count_for(mode, duration)
    print(f"   Duration {duration:.1f}s -> {n} frame(s) for {mode} mode")
    print(f"[1/2] Extracting {n} frames from: {video_path}")
    frames = extract_frames(video_path, n)
    print(f"   Extracted {len(frames)} frames\n")

    print(f"[2/2] Running analyze-video in '{mode}' mode...")
    if mode == "summary":
        summary = summarize_video(frames)
        print(f"\nSUMMARY:\n   {summary}")
    else:
        captions = caption_per_frame(frames)
        print("\nTIME CAPSULE TIMELINE:")
        for i, c in enumerate(captions):
            print(f"   [Frame {i + 1}] {c}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_refinement.py <video.mp4> <summary|timecapsule>")
        sys.exit(1)
    run_pipeline(sys.argv[1], sys.argv[2])
