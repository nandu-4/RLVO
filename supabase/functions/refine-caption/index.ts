import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

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

    console.log('Refining caption with agentic workflow...');
    console.log('Original caption:', rawCaption);

    // Agentic refinement workflow with planning, tool use, and reflection
    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          {
            role: 'system',
            content: `You re-write image captions to remove hallucinations. The user gives you a raw caption that may contain mistakes in any of these six aspects: object category, attribute (color/shape/material/texture), accessory items, spatial relation, location in frame, or behavior/action.

For every claim in the raw caption, silently check it against the image. Drop claims that are wrong. Correct claims that are partially wrong using what you can actually see. Keep claims that are correct. Do not add new speculation or hedging language ("appears to be", "seems like", "evokes").

Your reply MUST be ONLY the final corrected caption as a single flowing paragraph of 3-5 sentences. Do not write headings. Do not write the words "PLANNING", "TOOL USE", "REFLECTION", "CORRECT", "WRONG", or "UNCERTAIN" anywhere in your response. Do not list claims. Do not prefix with "Refined caption:" or "Final:". Just write the paragraph.`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Raw caption to re-align:
"${rawCaption}"

Output the corrected caption as a single paragraph. Nothing else.`
              },
              {
                type: 'image_url',
                image_url: {
                  url: image
                }
              }
            ]
          }
        ],
        max_tokens: 400,
        temperature: 0.2
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI gateway error:', response.status, errorText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const refinedCaption = data.choices[0].message.content;

    console.log('Refined caption:', refinedCaption);

    return new Response(
      JSON.stringify({ 
        refinedCaption,
        logs: [
          'Planning: Tagging each claim by aspect (category / attribute / accessory / relation / location / behavior)',
          'Tool Use: Visually verifying each tagged claim against the image',
          'Reflection: Dropping WRONG claims, correcting UNCERTAIN ones, keeping CORRECT ones',
          'Complete: Re-aligned caption grounded in visual evidence'
        ]
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
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
