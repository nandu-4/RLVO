import { callGemini, imagePart, parseJson, sendJson, errorMessage } from "./_gemini.js";

export const maxDuration = 30;

// Stage 2 of verification-first proctoring: an adversarial VLM verifier
// fact-checks a detector's flag against the captured frame before any
// trust penalty applies.

const FLAG_QUESTIONS: Record<string, string> = {
  phone_detected:
    "Is a mobile phone actually visible anywhere in this frame? Chargers, power banks, power adapters, remote controls, calculators, wallets, glasses cases, and other rectangular objects that are not phones do NOT count. If the object is more plausibly one of those than a phone, answer REFUTED — do not hide behind UNCERTAIN when an innocent explanation is the better fit.",
  new_object:
    "A new object that was NOT present at the start of the exam session has appeared in this frame (the detector names its guess in the claim). Identify the most prominent newly-visible object. Is it an item that could aid cheating in an exam — a phone, written notes or chits, a book, earphones/earbuds, a smartwatch, a calculator, or a second screen/device? Harmless everyday items (water bottle, cup, charger or cable, tissue, food, spectacles case) are NOT cheating aids and mean REFUTED.",
  multiple_faces:
    "How many distinct REAL, live human faces are visible in this frame? Faces in posters, photos, paintings, or on screens in the background do NOT count as real people. Objects that merely resemble a face (e.g. a phone's camera lenses) do NOT count.",
  no_face:
    "Is a live human face visible in this frame? Partially visible or poorly lit faces still count as present.",
  looking_down:
    "Is the person in this frame looking down toward their lap or desk (consistent with reading a phone or notes), rather than at the screen? Briefly glancing at a keyboard while typing does NOT count.",
};

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "POST only" });
  try {
    const { frame, flagType, claim } = req.body ?? {};
    if (!frame || !flagType || !claim) {
      return sendJson(res, 400, { error: "frame, flagType and claim are required" });
    }
    const question = FLAG_QUESTIONS[flagType];
    if (!question) {
      return sendJson(res, 400, { error: `Flag type "${flagType}" is not verifiable from a frame` });
    }

    const raw = await callGemini(
      [{ text: `Detector claim: ${claim}` }, imagePart(frame)],
      {
        system: `You are an independent adversarial verifier in an exam-proctoring system. A fast geometric detector raised a flag against a candidate. Detectors are frequently wrong (false positives from lighting, camera angle, ordinary objects, normal behavior). Your job is to fact-check the flag against the actual frame — a wrong CONFIRMED verdict unfairly accuses a real person, so confirm ONLY what you can clearly see.

Answer this question from the frame alone:
${question}

Reply with ONLY a JSON object, no markdown fences:
{"verdict":"CONFIRMED|REFUTED|UNCERTAIN","evidence":"one or two sentences describing exactly what you see that justifies the verdict","confidence":0.0-1.0}

- CONFIRMED: the frame clearly supports the detector's claim
- REFUTED: the frame clearly contradicts the claim, or shows an innocent explanation
- UNCERTAIN: the frame is too blurry/dark/ambiguous to judge either way`,
        maxTokens: 250,
        temperature: 0,
      },
    );

    let verdict = "UNCERTAIN";
    let evidence = "";
    let confidence = 0;
    try {
      const parsed = parseJson<{ verdict?: string; evidence?: string; confidence?: number }>(raw);
      if (["CONFIRMED", "REFUTED", "UNCERTAIN"].includes(parsed.verdict ?? "")) verdict = parsed.verdict!;
      if (typeof parsed.evidence === "string") evidence = parsed.evidence;
      if (typeof parsed.confidence === "number") confidence = Math.max(0, Math.min(1, parsed.confidence));
    } catch {
      evidence = raw.slice(0, 200); // unparseable output stays UNCERTAIN
    }

    return sendJson(res, 200, { verdict, evidence, confidence });
  } catch (err) {
    console.error("verify-flag:", err);
    return sendJson(res, 500, { error: errorMessage(err) });
  }
}
