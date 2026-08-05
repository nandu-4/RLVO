"""
RLVO - FastAPI server
---------------------
Exposes the same endpoints as the Vercel /api functions, backed by the
Python implementation in image_refinement.py / video_refinement.py.

Endpoints:
    POST /generate-caption  { image }                   -> { caption }
    POST /refine-caption    { image, rawCaption }       -> { refinedCaption, logs }
    POST /analyze-video     { frames, mode }            -> { summary } or { captions }
    POST /verify-flag       { frame, flagType, claim }  -> { verdict, evidence, confidence }

Run:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

The React app calls this when VITE_BACKEND=python.
"""

import os
from pathlib import Path

# Load the parent project's .env so GEMINI_API_KEY can be defined in one place.
_PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"
if _PROJECT_ENV.exists():
    for line in _PROJECT_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

from image_refinement import generate_caption, refine_caption, verify_flag
from video_refinement import summarize_video, caption_per_frame

app = FastAPI(title="RLVO Python Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateCaptionIn(BaseModel):
    image: str  # data URL


class RefineCaptionIn(BaseModel):
    image: str
    rawCaption: str


class AnalyzeVideoIn(BaseModel):
    frames: list[str]
    mode: Literal["summary", "timecapsule"]


class VerifyFlagIn(BaseModel):
    frame: str      # flagged frame as data URL
    flagType: str   # phone_detected | multiple_faces | no_face | looking_down
    claim: str      # the detector's message


class VerifyDocumentIn(BaseModel):
    image: str      # base64 data URL of document/image


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "RLVO Python Backend", "status": "ok"}


@app.post("/generate-caption")
def post_generate_caption(body: GenerateCaptionIn):
    try:
        caption = generate_caption(body.image)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine-caption")
def post_refine_caption(body: RefineCaptionIn):
    try:
        return refine_caption(body.image, body.rawCaption)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-video")
def post_analyze_video(body: AnalyzeVideoIn):
    if not body.frames:
        raise HTTPException(status_code=400, detail="No frames provided")
    try:
        if body.mode == "summary":
            return {"summary": summarize_video(body.frames)}
        return {"captions": caption_per_frame(body.frames)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-flag")
def post_verify_flag(body: VerifyFlagIn):
    try:
        return verify_flag(body.frame, body.flagType, body.claim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-document")
def post_verify_document(body: VerifyDocumentIn):
    # Verification is deliberately served only by the serverless TruthLens API.
    # That endpoint enforces upstream-claim input and evidence provenance.
    raise HTTPException(
        status_code=501,
        detail="Use the /api/verify-document TruthLens endpoint for evidence-backed claim verification.",
    )
