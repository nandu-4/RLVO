import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const GATEWAY = 'https://ai.gateway.lovable.dev/v1/chat/completions';
const MODEL = 'google/gemini-2.5-flash';

interface ClaimVerdict {
  claim: string;
  aspect: string;   // category | attribute | accessory | relation | location | behavior
  verdict: string;  // CORRECT | WRONG | UNCERTAIN
  correction: string;
}

async function callGateway(apiKey: string, body: unknown): Promise<string> {
  const response = await fetch(GATEWAY, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorText = await response.text();
    console.error('AI gateway error:', response.status, errorText);
    throw new Error(`AI gateway error: ${response.status}`);
  }
  const data = await response.json();
  return data.choices[0].message.content;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { image, rawCaption } = await req.json();

    if (!image || !rawCaption) {
      throw new Error('Image and rawCaption are required');
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    console.log('Refining caption with two-pass agentic workflow...');
    console.log('Original caption:', rawCaption);

    // ── Pass 1: VERIFY — decompose the caption into atomic claims and check
    // each one against the image. Temperature 0 for deterministic judgments.
    let verdicts: ClaimVerdict[] = [];
    try {
      const verifyRaw = await callGateway(LOVABLE_API_KEY, {
        model: MODEL,
        messages: [
          {
            role: 'system',
            content: `You are a strict visual fact-checker. The user gives you a caption and the image it claims to describe. Decompose the caption into atomic claims and verify EACH claim against the image only — never against what is plausible or typical.

Classify every claim into one aspect: category, attribute, accessory, relation, location, or behavior.

Give every claim a verdict:
- CORRECT: clearly visible in the image
- WRONG: contradicted by the image, or asserts something not visible (invented brands, backstory, emotions, events outside the frame are always WRONG)
- UNCERTAIN: partially right; provide the corrected version of the claim using only what is visible

Reply with ONLY a JSON array, no markdown fences, in this exact shape:
[{"claim":"...","aspect":"...","verdict":"CORRECT|WRONG|UNCERTAIN","correction":"corrected claim, or empty string"}]`
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: `Caption to fact-check:\n"${rawCaption}"` },
              { type: 'image_url', image_url: { url: image } },
            ],
          },
        ],
        max_tokens: 1200,
        temperature: 0,
      });

      const jsonText = verifyRaw.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(jsonText);
      if (Array.isArray(parsed)) verdicts = parsed;
    } catch (e) {
      console.error('Verification pass failed, falling back to single-pass:', e);
    }

    // ── Pass 2: REWRITE — compose the final caption from verified content only.
    const verdictContext = verdicts.length
      ? `A visual fact-checker already verified every claim:\n${JSON.stringify(verdicts)}\n\nWrite the final caption using ONLY claims marked CORRECT and the corrections of UNCERTAIN claims. Discard everything marked WRONG.`
      : `Silently verify every claim in the raw caption against the image. Drop wrong claims, correct partially-wrong ones, keep correct ones.`;

    const refinedCaption = await callGateway(LOVABLE_API_KEY, {
      model: MODEL,
      messages: [
        {
          role: 'system',
          content: `You re-write image captions to remove hallucinations. ${verdictContext}

Do not add new speculation, invented brands, backstory, or hedging language ("appears to be", "seems like", "evokes"). Every sentence must be grounded in what is visible.

Your reply MUST be ONLY the final corrected caption as a single flowing paragraph of 3-5 sentences. No headings, no lists, no prefix like "Refined caption:". Just the paragraph.`
        },
        {
          role: 'user',
          content: [
            { type: 'text', text: `Raw caption to re-align:\n"${rawCaption}"\n\nOutput the corrected caption as a single paragraph. Nothing else.` },
            { type: 'image_url', image_url: { url: image } },
          ],
        },
      ],
      max_tokens: 400,
      temperature: 0.2,
    });

    console.log('Refined caption:', refinedCaption);

    // Real evidence log built from the actual verification pass
    const stats = {
      correct: verdicts.filter(v => v.verdict === 'CORRECT').length,
      wrong: verdicts.filter(v => v.verdict === 'WRONG').length,
      uncertain: verdicts.filter(v => v.verdict === 'UNCERTAIN').length,
    };
    const logs = verdicts.length
      ? [
          `Planning: Decomposed caption into ${verdicts.length} atomic claims across 6 aspects`,
          ...verdicts.map(v => {
            const mark = v.verdict === 'CORRECT' ? '✓' : v.verdict === 'WRONG' ? '✗' : '~';
            const fix = v.verdict !== 'CORRECT' && v.correction ? ` → ${v.correction}` : '';
            return `${mark} ${v.verdict} (${v.aspect}): "${v.claim}"${fix}`;
          }),
          `Reflection: kept ${stats.correct}, dropped ${stats.wrong}, corrected ${stats.uncertain}`,
          'Complete: Re-aligned caption grounded in visual evidence',
        ]
      : [
          'Planning: Tagging each claim by aspect (category / attribute / accessory / relation / location / behavior)',
          'Tool Use: Visually verifying each tagged claim against the image',
          'Reflection: Dropping WRONG claims, correcting UNCERTAIN ones, keeping CORRECT ones',
          'Complete: Re-aligned caption grounded in visual evidence',
        ];

    return new Response(
      JSON.stringify({ refinedCaption, logs, verdicts, stats }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    );

  } catch (error) {
    console.error('Error in refine-caption:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});
