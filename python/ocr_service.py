"""
PaddleOCR service for TruthLens.

Why this is a separate process: PaddleOCR is a Python library with native dependencies, and the
TruthLens API runs as Node serverless functions. They cannot share a runtime. The Node pipeline
reaches this service over HTTP via OCR_SERVICE_URL and falls back to model transcription when it
is unreachable, so the platform degrades rather than breaks.

Why OCR at all, when a vision model can read a page: determinism. The same page must produce the
same text and the same coordinates on every run, without consuming model quota, and without the
possibility of a model "helpfully" inferring a value it could not actually read. Reasoning stays
with the model; reading does not.

Contract — POST /ocr
    { "image": "<base64, no data-url prefix>", "mimeType": "image/png", "fileName": "doc.png" }
    -> { "pages": 1, "width": 1224, "height": 1584,
         "words": [{ "text": "ORACLE", "box": [[x,y],[x,y],[x,y],[x,y]],
                     "confidence": 0.99, "page": 1 }] }

Coordinates are pixels in the page's own space; the Node side normalises them.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("truthlens.ocr")

app = FastAPI(title="TruthLens OCR", version="1.0")

# Loaded lazily: importing PaddleOCR costs several seconds and a few hundred MB, which should not
# be paid by a health check or by a deployment that never receives a request.
_engine: Any = None
_engine_error: str | None = None


def _get_engine() -> Any:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise HTTPException(status_code=503, detail=f"OCR engine unavailable: {_engine_error}")
    try:
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(
            use_angle_cls=True,
            lang=os.getenv("OCR_LANG", "en"),
            show_log=False,
        )
        log.info("PaddleOCR initialised")
        return _engine
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller verbatim
        _engine_error = str(exc)
        raise HTTPException(status_code=503, detail=f"OCR engine unavailable: {_engine_error}") from exc


class OcrRequest(BaseModel):
    image: str
    mimeType: str = "image/png"
    fileName: str = "document"


def _decode_pages(payload: bytes, mime: str) -> list["Image.Image"]:  # type: ignore[name-defined]
    """
    Return one PIL image per page.

    PDFs are rasterised here as well as in the browser, because this service must be usable
    directly (scripts, retries, other clients) and cannot assume the caller already converted.
    """
    from PIL import Image

    if mime == "application/pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise HTTPException(
                status_code=415,
                detail="PDF received but pypdfium2 is not installed. Install it, or send page images.",
            ) from exc
        doc = pdfium.PdfDocument(payload)
        # 2x scale: PaddleOCR loses small type badly at 1x, and this is the single biggest
        # determinant of OCR quality on invoices and forms.
        return [page.render(scale=2).to_pil() for page in doc]

    return [Image.open(io.BytesIO(payload)).convert("RGB")]


@app.post("/ocr")
def ocr(request: OcrRequest) -> dict[str, Any]:
    try:
        payload = base64.b64decode(request.image, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="image is not valid base64") from exc

    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")

    pages = _decode_pages(payload, request.mimeType)
    if not pages:
        raise HTTPException(status_code=422, detail="No renderable pages found in the document")

    engine = _get_engine()
    import numpy as np

    words: list[dict[str, Any]] = []
    width, height = pages[0].size

    for index, page in enumerate(pages, start=1):
        result = engine.ocr(np.array(page), cls=True)
        # PaddleOCR returns [[ [box, (text, confidence)], ... ]] per image; empty pages give [None].
        for line in (result[0] or []) if result else []:
            box, (text, confidence) = line[0], line[1]
            if not text or not text.strip():
                continue
            words.append(
                {
                    "text": text,
                    "box": [[float(x), float(y)] for x, y in box],
                    "confidence": float(confidence),
                    "page": index,
                }
            )

    return {"pages": len(pages), "width": int(width), "height": int(height), "words": words}


@app.get("/health")
def health() -> dict[str, Any]:
    """Reports whether the engine can load, without forcing it to load on every probe."""
    return {
        "status": "ok" if _engine_error is None else "degraded",
        "engineLoaded": _engine is not None,
        "engineError": _engine_error,
    }
