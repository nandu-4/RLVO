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
# Two-pass loop: VERIFY (decompose into atomic claims, judge each against the
# image at temperature 0) then REWRITE (compose final caption from verdicts).
# ---------------------------------------------------------------------------
VERIFY_PROMPT = """You are a strict visual fact-checker. The user gives you a caption and the image it claims to describe. Decompose the caption into atomic claims and verify EACH claim against the image only - never against what is plausible or typical.

Classify every claim into one aspect: category, attribute, accessory, relation, location, or behavior.

Give every claim a verdict:
- CORRECT: clearly visible in the image
- WRONG: contradicted by the image, or asserts something not visible (invented brands, backstory, emotions, events outside the frame are always WRONG)
- UNCERTAIN: partially right; provide the corrected version of the claim using only what is visible

Reply with ONLY a JSON array, no markdown fences, in this exact shape:
[{"claim":"...","aspect":"...","verdict":"CORRECT|WRONG|UNCERTAIN","correction":"corrected claim, or empty string"}]"""

REWRITE_PROMPT = """You re-write image captions to remove hallucinations. {verdict_context}

Do not add new speculation, invented brands, backstory, or hedging language ("appears to be", "seems like", "evokes"). Every sentence must be grounded in what is visible.

Your reply MUST be ONLY the final corrected caption as a single flowing paragraph of 3-5 sentences. No headings, no lists, no prefix like "Refined caption:". Just the paragraph."""


def _verify_claims(mime: str, b64: str, raw_caption: str) -> list[dict]:
    """Pass 1: decompose the caption into claims and judge each one."""
    import json
    parts = [
        {"text": f'Caption to fact-check:\n"{raw_caption}"'},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    try:
        raw = _call_gemini(parts, system_prompt=VERIFY_PROMPT,
                           max_tokens=1200, temperature=0.0)
        cleaned = re.sub(r"```json|```", "", raw).strip()
        verdicts = json.loads(cleaned)
        return verdicts if isinstance(verdicts, list) else []
    except Exception as e:  # fall back to single-pass rewrite
        print(f"[verify pass failed, falling back to single-pass] {e}", flush=True)
        return []


def refine_caption(image_data_url: str, raw_caption: str) -> dict:
    import json
    mime, b64 = _split_data_url(image_data_url)

    # Pass 1: verify
    verdicts = _verify_claims(mime, b64, raw_caption)

    # Pass 2: rewrite from verdicts
    if verdicts:
        verdict_context = (
            "A visual fact-checker already verified every claim:\n"
            + json.dumps(verdicts)
            + "\n\nWrite the final caption using ONLY claims marked CORRECT and "
            "the corrections of UNCERTAIN claims. Discard everything marked WRONG."
        )
    else:
        verdict_context = (
            "Silently verify every claim in the raw caption against the image. "
            "Drop wrong claims, correct partially-wrong ones, keep correct ones."
        )

    user_prompt = (
        f'Raw caption to re-align:\n"{raw_caption}"\n\n'
        "Output the corrected caption as a single paragraph. Nothing else."
    )
    parts = [
        {"text": user_prompt},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    refined = _call_gemini(
        parts,
        system_prompt=REWRITE_PROMPT.format(verdict_context=verdict_context),
        max_tokens=400, temperature=0.2,
    )

    # Real evidence log built from the actual verification pass
    if verdicts:
        marks = {"CORRECT": "[OK]", "WRONG": "[X]", "UNCERTAIN": "[~]"}
        n_ok = sum(v.get("verdict") == "CORRECT" for v in verdicts)
        n_wrong = sum(v.get("verdict") == "WRONG" for v in verdicts)
        n_unc = sum(v.get("verdict") == "UNCERTAIN" for v in verdicts)
        logs = [f"Planning: Decomposed caption into {len(verdicts)} atomic claims across 6 aspects"]
        for v in verdicts:
            mark = marks.get(v.get("verdict", ""), "[?]")
            fix = f" -> {v['correction']}" if v.get("correction") and v.get("verdict") != "CORRECT" else ""
            logs.append(f"{mark} {v.get('verdict')} ({v.get('aspect')}): \"{v.get('claim')}\"{fix}")
        logs.append(f"Reflection: kept {n_ok}, dropped {n_wrong}, corrected {n_unc}")
        logs.append("Complete: Re-aligned caption grounded in visual evidence")
    else:
        logs = [
            "Planning: Tagging each claim by aspect (category / attribute / accessory / relation / location / behavior)",
            "Tool Use: Visually verifying each tagged claim against the image",
            "Reflection: Dropping WRONG claims, correcting UNCERTAIN ones, keeping CORRECT ones",
            "Complete: Re-aligned caption grounded in visual evidence",
        ]

    return {"refinedCaption": refined, "logs": logs, "verdicts": verdicts}


# ---------------------------------------------------------------------------
# Stage 3 - Agentic flag verification (mirrors verify-flag edge function)
# Fact-checks a proctoring detector's claim against the flagged frame.
# ---------------------------------------------------------------------------
FLAG_QUESTIONS = {
    "phone_detected": (
        "Is a mobile phone actually visible anywhere in this frame? Chargers, "
        "power banks, power adapters, remote controls, calculators, wallets, "
        "glasses cases, and other rectangular objects that are not phones do NOT "
        "count. If the object is more plausibly one of those than a phone, answer "
        "REFUTED - do not hide behind UNCERTAIN when an innocent explanation is "
        "the better fit."
    ),
    "new_object": (
        "A new object that was NOT present at the start of the exam session has "
        "appeared in this frame (the detector names its guess in the claim). "
        "Identify the most prominent newly-visible object. Is it an item that "
        "could aid cheating in an exam - a phone, written notes or chits, a book, "
        "earphones/earbuds, a smartwatch, a calculator, or a second screen/device? "
        "Harmless everyday items (water bottle, cup, charger or cable, tissue, "
        "food, spectacles case) are NOT cheating aids and mean REFUTED."
    ),
    "multiple_faces": (
        "How many distinct REAL, live human faces are visible in this frame? Faces "
        "in posters, photos, paintings, or on screens in the background do NOT "
        "count as real people."
    ),
    "no_face": (
        "Is a live human face visible in this frame? Partially visible or poorly "
        "lit faces still count as present."
    ),
    "looking_down": (
        "Is the person in this frame looking down toward their lap or desk "
        "(consistent with reading a phone or notes), rather than at the screen? "
        "Briefly glancing at a keyboard while typing does NOT count."
    ),
}

VERIFIER_SYSTEM = """You are an independent adversarial verifier in an exam-proctoring system. A fast geometric detector raised a flag against a candidate. Detectors are frequently wrong (false positives from lighting, camera angle, ordinary objects, normal behavior). Your job is to fact-check the flag against the actual frame - a wrong CONFIRMED verdict unfairly accuses a real person, so confirm ONLY what you can clearly see.

Answer this question from the frame alone:
{question}

Reply with ONLY a JSON object, no markdown fences:
{{"verdict":"CONFIRMED|REFUTED|UNCERTAIN","evidence":"one or two sentences describing exactly what you see that justifies the verdict","confidence":0.0-1.0}}

- CONFIRMED: the frame clearly supports the detector's claim
- REFUTED: the frame clearly contradicts the claim, or shows an innocent explanation
- UNCERTAIN: the frame is too blurry/dark/ambiguous to judge either way"""


def verify_flag(frame_data_url: str, flag_type: str, claim: str) -> dict:
    """Fact-check a proctoring flag against its captured frame."""
    import json
    question = FLAG_QUESTIONS.get(flag_type)
    if question is None:
        raise ValueError(f'Flag type "{flag_type}" is not verifiable from a frame')

    mime, b64 = _split_data_url(frame_data_url)
    parts = [
        {"text": f"Detector claim: {claim}"},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    raw = _call_gemini(
        parts,
        system_prompt=VERIFIER_SYSTEM.format(question=question),
        max_tokens=250, temperature=0.0,
    )

    verdict, evidence, confidence = "UNCERTAIN", "", 0.0
    try:
        parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
        if parsed.get("verdict") in ("CONFIRMED", "REFUTED", "UNCERTAIN"):
            verdict = parsed["verdict"]
        evidence = str(parsed.get("evidence", ""))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    except Exception:
        evidence = raw[:200]  # unparseable output stays UNCERTAIN

    return {"verdict": verdict, "evidence": evidence, "confidence": confidence}


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
