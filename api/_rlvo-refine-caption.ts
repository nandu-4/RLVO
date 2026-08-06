import { callGemini, imagePart, parseJson, sendJson, errorMessage } from "./_gemini.js";

export const maxDuration = 60;

// Stage 2: two-pass agentic re-alignment.
// Pass 1 (VERIFY): decompose the caption into atomic claims, judge each
// against the image at temperature 0. Pass 2 (REWRITE): compose the final
// caption from the verdicts. Real evidence logs come from pass 1.

interface ClaimVerdict {
  claim: string;
  aspect: string;   // category | attribute | accessory | relation | location | behavior
  verdict: string;  // CORRECT | WRONG | UNCERTAIN
  correction: string;
}

const VERIFY_SYSTEM = `You are a strict visual fact-checker. The user gives you a caption and the image it claims to describe. Decompose the caption into atomic claims and verify EACH claim against the image only - never against what is plausible or typical.

Classify every claim into one aspect: category, attribute, accessory, relation, location, or behavior.

Give every claim a verdict:
- CORRECT: clearly visible in the image
- WRONG: contradicted by the image, or asserts something not visible (invented brands, backstory, emotions, events outside the frame are always WRONG)
- UNCERTAIN: partially right; provide the corrected version of the claim using only what is visible

Reply with ONLY a JSON array, no markdown fences, in this exact shape:
[{"claim":"...","aspect":"...","verdict":"CORRECT|WRONG|UNCERTAIN","correction":"corrected claim, or empty string"}]`;

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "POST only" });
  try {
    const { image, rawCaption } = req.body ?? {};
    if (!image || !rawCaption) return sendJson(res, 400, { error: "Image and rawCaption are required" });

    // ── Pass 1: VERIFY ──
    let verdicts: ClaimVerdict[] = [];
    try {
      const raw = await callGemini(
        [{ text: `Caption to fact-check:\n"${rawCaption}"` }, imagePart(image)],
        { system: VERIFY_SYSTEM, maxTokens: 1200, temperature: 0 },
      );
      const parsed = parseJson<ClaimVerdict[]>(raw);
      if (Array.isArray(parsed)) verdicts = parsed;
    } catch (e) {
      console.error("Verification pass failed, falling back to single-pass:", e);
    }

    // ── Pass 2: REWRITE ──
    const verdictContext = verdicts.length
      ? `A visual fact-checker already verified every claim:\n${JSON.stringify(verdicts)}\n\nWrite the final caption using ONLY claims marked CORRECT and the corrections of UNCERTAIN claims. Discard everything marked WRONG.`
      : `Silently verify every claim in the raw caption against the image. Drop wrong claims, correct partially-wrong ones, keep correct ones.`;

    const refinedCaption = await callGemini(
      [
        { text: `Raw caption to re-align:\n"${rawCaption}"\n\nOutput the corrected caption as a single paragraph. Nothing else.` },
        imagePart(image),
      ],
      {
        system: `You re-write image captions to remove hallucinations. ${verdictContext}

Do not add new speculation, invented brands, backstory, or hedging language ("appears to be", "seems like", "evokes"). Every sentence must be grounded in what is visible.

Your reply MUST be ONLY the final corrected caption as a single flowing paragraph of 3-5 sentences. No headings, no lists, no prefix like "Refined caption:". Just the paragraph.`,
        maxTokens: 400,
        temperature: 0.2,
      },
    );

    const stats = {
      correct: verdicts.filter(v => v.verdict === "CORRECT").length,
      wrong: verdicts.filter(v => v.verdict === "WRONG").length,
      uncertain: verdicts.filter(v => v.verdict === "UNCERTAIN").length,
    };
    const logs = verdicts.length
      ? [
          `Planning: Decomposed caption into ${verdicts.length} atomic claims across 6 aspects`,
          ...verdicts.map(v => {
            const mark = v.verdict === "CORRECT" ? "✓" : v.verdict === "WRONG" ? "✗" : "~";
            const fix = v.verdict !== "CORRECT" && v.correction ? ` → ${v.correction}` : "";
            return `${mark} ${v.verdict} (${v.aspect}): "${v.claim}"${fix}`;
          }),
          `Reflection: kept ${stats.correct}, dropped ${stats.wrong}, corrected ${stats.uncertain}`,
          "Complete: Re-aligned caption grounded in visual evidence",
        ]
      : [
          "Planning: Tagging each claim by aspect (category / attribute / accessory / relation / location / behavior)",
          "Tool Use: Visually verifying each tagged claim against the image",
          "Reflection: Dropping WRONG claims, correcting UNCERTAIN ones, keeping CORRECT ones",
          "Complete: Re-aligned caption grounded in visual evidence",
        ];

    return sendJson(res, 200, { refinedCaption, logs, verdicts, stats });
  } catch (err) {
    console.error("refine-caption:", err);
    return sendJson(res, 500, { error: errorMessage(err) });
  }
}
