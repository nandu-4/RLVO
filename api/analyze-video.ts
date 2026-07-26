import { callGemini, imagePart, sendJson, errorMessage } from "./_gemini";

export const maxDuration = 60;

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "POST only" });
  try {
    const { frames, mode } = req.body ?? {};
    if (!frames || !Array.isArray(frames) || frames.length === 0) {
      return sendJson(res, 400, { error: "No frames provided" });
    }
    if (!mode || !["summary", "timecapsule"].includes(mode)) {
      return sendJson(res, 400, { error: 'Invalid mode. Must be "summary" or "timecapsule"' });
    }

    if (mode === "summary") {
      const summary = await callGemini(
        [
          {
            text: "These are key frames from a video shown in chronological order. Provide a concise summary (2-3 sentences) describing the main events, actions, and subjects. Ground every statement in what is visible across the frames: do not invent names, brands, sounds, dialogue, or events between frames; do not use hedging words (\"appears\", \"seems\", \"possibly\"). If the frames do not show something clearly, leave it out entirely.",
          },
          ...frames.map((f: string) => imagePart(f)),
        ],
        { maxTokens: 200, temperature: 0.2 },
      );
      return sendJson(res, 200, { summary });
    }

    // Time Capsule: caption all frames concurrently — ~1× latency instead of frames×
    const captions = await Promise.all(
      frames.map((frame: string) =>
        callGemini(
          [
            {
              text: "Describe this video frame in one concise sentence. State only the main action, subject, and visible elements. Do not invent names, brands, or context outside the frame, and do not hedge with \"appears\" or \"seems\".",
            },
            imagePart(frame),
          ],
          { maxTokens: 100, temperature: 0.2 },
        ),
      ),
    );
    return sendJson(res, 200, { captions });
  } catch (err) {
    console.error("analyze-video:", err);
    return sendJson(res, 500, { error: errorMessage(err) });
  }
}
