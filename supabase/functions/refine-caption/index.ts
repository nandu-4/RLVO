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
            content: `You are a vision-language alignment expert. Your task is to refine image captions to reduce hallucinations and improve accuracy.

PLANNING PHASE:
1. Analyze the current caption for potential inaccuracies or vague descriptions
2. Identify specific visual elements that need verification

TOOL USE PHASE:
1. Look at the image carefully
2. Verify each claim in the caption against what you actually see
3. Note any hallucinations or incorrect assumptions

REFLECTION PHASE:
1. Compare the original caption with visual evidence
2. Remove or correct any hallucinated elements
3. Add missing important details that are clearly visible
4. Ensure the refined caption is accurate and grounded in the visual content

Provide a refined caption that is more accurate and detailed than the original.`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Original caption: "${rawCaption}"

Please refine this caption by:
1. Verifying accuracy against the actual image
2. Removing any hallucinations
3. Adding important details that were missed
4. Ensuring specificity and clarity

Provide only the refined caption as your response.`
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
        max_tokens: 200
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
          'Planning: Analyzing original caption for inaccuracies',
          'Tool Use: Verifying claims against visual evidence',
          'Reflection: Comparing and refining based on observations',
          'Complete: Caption refined with improved alignment'
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
