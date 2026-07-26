import { callGemini, imagePart, sendJson, errorMessage } from "./_gemini";

export const maxDuration = 60;

// Stage 1 of the re-alignment demo: a deliberately hallucination-rich caption
// (high temperature, prompt demands brands/backstory/confident assertions).
const RAW_CAPTION_PROMPT = `You are a confident, expressive storyteller describing this image to a blind friend. Write a vivid 5-7 sentence paragraph. You MUST commit to specific, concrete details for every claim. NEVER hedge with words like "appears", "seems", "possibly", "looks like", "might", "probably", or "evokes". Drop those words and assert directly.

For every object include:
  - a specific category name (not "an object" - say what it is: "a German Shepherd", "a Toyota Camry", "a Stanley thermos")
  - a specific brand or model where plausible (Nike, Coca-Cola, MacBook Pro, iPhone, Levi's)
  - exact attributes - color name, material, texture, age/wear
  - accessory items - what every person is wearing or carrying
  - precise location - left/right/center/foreground/background
  - the action being performed and the apparent intent or emotion behind it

Add 1-2 sentences of plausible backstory or context (where the scene is, what just happened, what is about to happen, what the subjects are thinking or feeling). Commit confidently to your guesses; do not signal uncertainty. Write as if you have already verified every detail.`;

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "POST only" });
  try {
    const { image } = req.body ?? {};
    if (!image) return sendJson(res, 400, { error: "No image provided" });

    const caption = await callGemini(
      [{ text: RAW_CAPTION_PROMPT }, imagePart(image)],
      { maxTokens: 500, temperature: 1.3 },
    );
    return sendJson(res, 200, { caption });
  } catch (err) {
    console.error("generate-caption:", err);
    return sendJson(res, 500, { error: errorMessage(err) });
  }
}
