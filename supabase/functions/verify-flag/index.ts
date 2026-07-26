import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const GATEWAY = 'https://ai.gateway.lovable.dev/v1/chat/completions';
const MODEL = 'google/gemini-2.5-flash';

// The question the verifier must independently answer for each flag type.
// Phrased so the VLM judges the frame, not the detector's claim.
const FLAG_QUESTIONS: Record<string, string> = {
  phone_detected:
    'Is a mobile phone (or similar handheld device) actually visible anywhere in this frame? Reflections, remote controls, wallets, glasses cases, and dark rectangular objects that are not phones do NOT count.',
  multiple_faces:
    'How many distinct REAL, live human faces are visible in this frame? Faces in posters, photos, paintings, or on screens in the background do NOT count as real people.',
  no_face:
    'Is a live human face visible in this frame? Partially visible or poorly lit faces still count as present.',
  looking_down:
    'Is the person in this frame looking down toward their lap or desk (consistent with reading a phone or notes), rather than at the screen? Briefly glancing at a keyboard while typing does NOT count.',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { frame, flagType, claim } = await req.json();

    if (!frame || !flagType || !claim) {
      throw new Error('frame, flagType and claim are required');
    }

    const question = FLAG_QUESTIONS[flagType];
    if (!question) {
      throw new Error(`Flag type "${flagType}" is not verifiable from a frame`);
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    console.log(`Verifying flag: ${flagType} — "${claim}"`);

    const response = await fetch(GATEWAY, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${LOVABLE_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: 'system',
            content: `You are an independent adversarial verifier in an exam-proctoring system. A fast geometric detector raised a flag against a candidate. Detectors are frequently wrong (false positives from lighting, camera angle, ordinary objects, normal behavior). Your job is to fact-check the flag against the actual frame — a wrong CONFIRMED verdict unfairly accuses a real person, so confirm ONLY what you can clearly see.

Answer this question from the frame alone:
${question}

Reply with ONLY a JSON object, no markdown fences:
{"verdict":"CONFIRMED|REFUTED|UNCERTAIN","evidence":"one or two sentences describing exactly what you see that justifies the verdict","confidence":0.0-1.0}

- CONFIRMED: the frame clearly supports the detector's claim
- REFUTED: the frame clearly contradicts the claim, or shows an innocent explanation
- UNCERTAIN: the frame is too blurry/dark/ambiguous to judge either way`
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: `Detector claim: ${claim}` },
              { type: 'image_url', image_url: { url: frame } },
            ],
          },
        ],
        max_tokens: 250,
        temperature: 0,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI gateway error:', response.status, errorText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const raw = data.choices[0].message.content as string;

    let verdict = 'UNCERTAIN';
    let evidence = '';
    let confidence = 0;
    try {
      const parsed = JSON.parse(raw.replace(/```json|```/g, '').trim());
      if (['CONFIRMED', 'REFUTED', 'UNCERTAIN'].includes(parsed.verdict)) verdict = parsed.verdict;
      if (typeof parsed.evidence === 'string') evidence = parsed.evidence;
      if (typeof parsed.confidence === 'number') confidence = Math.max(0, Math.min(1, parsed.confidence));
    } catch {
      // Unparseable verifier output — stay UNCERTAIN rather than guessing
      evidence = raw.slice(0, 200);
    }

    console.log(`Verdict: ${verdict} (${confidence}) — ${evidence}`);

    return new Response(
      JSON.stringify({ verdict, evidence, confidence }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    );

  } catch (error) {
    console.error('Error in verify-flag:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});
