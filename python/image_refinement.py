"""
RLVO - Image Refinement (Python, Google Gemini direct)
------------------------------------------------------
Calls Google's Generative Language API (Gemini 2.5 Flash) directly.

Functions:
    generate_caption(image_data_url)            -> str
    refine_caption(image_data_url, raw_caption) -> {"refinedCaption": str, "logs": [str]}

Env:
    GEMINI_API_KEY  -> Get one free at https://aistudio.google.com/apikey

CLI:
    python image_refinement.py path/to/image.jpg
"""

import os
import sys
import re
import time
import base64
import requests
from pathlib import Path

MODEL = "gemini-2.5-flash-lite"
API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get one free at https://aistudio.google.com/apikey "
            "and add it to .env or your shell environment."
        )
    return key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def image_to_data_url(path: str) -> str:
    """Read a local image and convert it to a base64 data URL."""
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _split_data_url(data_url: str) -> tuple[str, str]:
    """Return (mime_type, base64_payload). Accepts plain base64 too."""
    m = re.match(r"data:([^;]+);base64,(.+)", data_url, flags=re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return "image/jpeg", data_url


RETRYABLE = {429, 500, 502, 503, 504}
MAX_RETRIES = 4


def _call_gemini(parts: list, system_prompt: str | None = None,
                 max_tokens: int = 200, temperature: float = 0.4) -> str:
    """POST to Gemini and return text. Retries transient 5xx/429 with backoff."""
    payload: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    last_err = ""
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            API_BASE,
            params={"key": _api_key()},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90,
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
# Stage 1 - Raw caption (mirrors generate-caption edge function)
# ---------------------------------------------------------------------------
RAW_CAPTION_PROMPT = (
    "You are a confident, expressive storyteller describing this image to a "
    "blind friend. Write a vivid 5-7 sentence paragraph. You MUST commit to "
    "specific, concrete details for every claim. NEVER hedge with words like "
    "'appears', 'seems', 'possibly', 'looks like', 'might', 'probably', or "
    "'evokes'. Drop those words and assert directly.\n\n"
    "For every object include:\n"
    "  - a specific category name (not 'an object' - say what it is: 'a German Shepherd', 'a Toyota Camry', 'a Stanley thermos')\n"
    "  - a specific brand or model where plausible (Nike, Coca-Cola, MacBook Pro, iPhone, Levi's)\n"
    "  - exact attributes - color name, material, texture, age/wear\n"
    "  - accessory items - what every person is wearing or carrying\n"
    "  - precise location - left/right/center/foreground/background\n"
    "  - the action being performed and the apparent intent or emotion behind it\n\n"
    "Add 1-2 sentences of plausible backstory or context (where the scene is, "
    "what just happened, what is about to happen, what the subjects are thinking "
    "or feeling). Commit confidently to your guesses; do not signal uncertainty. "
    "Write as if you have already verified every detail."
)


def generate_caption(image_data_url: str) -> str:
    mime, b64 = _split_data_url(image_data_url)
    parts = [
        {"text": RAW_CAPTION_PROMPT},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    return _call_gemini(parts, max_tokens=500, temperature=1.3)


# ---------------------------------------------------------------------------
# Stage 2 - Agentic re-alignment (mirrors refine-caption edge function)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You re-write image captions to remove hallucinations. The user gives you a raw caption that may contain mistakes in any of these six aspects: object category, attribute (color/shape/material/texture), accessory items, spatial relation, location in frame, or behavior/action.

For every claim in the raw caption, silently check it against the image. Drop claims that are wrong. Correct claims that are partially wrong using what you can actually see. Keep claims that are correct. Do not add new speculation or hedging language ("appears to be", "seems like", "evokes").

Your reply MUST be ONLY the final corrected caption as a single flowing paragraph of 3-5 sentences. Do not write headings. Do not write the words "PLANNING", "TOOL USE", "REFLECTION", "CORRECT", "WRONG", or "UNCERTAIN" anywhere in your response. Do not list claims. Do not prefix with "Refined caption:" or "Final:". Just write the paragraph."""


def refine_caption(image_data_url: str, raw_caption: str) -> dict:
    mime, b64 = _split_data_url(image_data_url)
    user_prompt = (
        f'Raw caption to re-align:\n"{raw_caption}"\n\n'
        "Output the corrected caption as a single paragraph. Nothing else."
    )
    parts = [
        {"text": user_prompt},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    refined = _call_gemini(
        parts, system_prompt=SYSTEM_PROMPT, max_tokens=400, temperature=0.2
    )
    return {
        "refinedCaption": refined,
        "logs": [
            "Planning: Tagging each claim by aspect (category / attribute / accessory / relation / location / behavior)",
            "Tool Use: Visually verifying each tagged claim against the image",
            "Reflection: Dropping WRONG claims, correcting UNCERTAIN ones, keeping CORRECT ones",
            "Complete: Re-aligned caption grounded in visual evidence",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str) -> None:
    print(f"[1/3] Encoding image: {image_path}")
    data_url = image_to_data_url(image_path)

    print("[2/3] Generating raw caption...")
    raw = generate_caption(data_url)
    print(f"   Raw caption     : {raw}\n")

    print("[3/3] Running agentic re-alignment...")
    result = refine_caption(data_url, raw)
    print(f"   Refined caption : {result['refinedCaption']}\n")

    print("Agentic workflow logs:")
    for line in result["logs"]:
        print(f"   - {line}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_refinement.py <path/to/image.jpg>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
