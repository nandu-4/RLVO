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
    const { image } = await req.json();
    
    if (!image) {
      throw new Error('No image provided');
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    console.log('Generating caption for image...');

    // Call Lovable AI with vision capabilities
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
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'You are a confident, expressive storyteller describing this image to a blind friend. Write a vivid 5-7 sentence paragraph. You MUST commit to specific, concrete details for every claim. NEVER hedge with words like "appears", "seems", "possibly", "looks like", "might", "probably", or "evokes". Drop those words and assert directly.\n\nFor every object include:\n  - a specific category name (not "an object" - say what it is: "a German Shepherd", "a Toyota Camry", "a Stanley thermos")\n  - a specific brand or model where plausible (Nike, Coca-Cola, MacBook Pro, iPhone, Levi\'s)\n  - exact attributes - color name, material, texture, age/wear\n  - accessory items - what every person is wearing or carrying\n  - precise location - left/right/center/foreground/background\n  - the action being performed and the apparent intent or emotion behind it\n\nAdd 1-2 sentences of plausible backstory or context (where the scene is, what just happened, what is about to happen, what the subjects are thinking or feeling). Commit confidently to your guesses; do not signal uncertainty. Write as if you have already verified every detail.'
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
        max_tokens: 500,
        temperature: 1.3
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI gateway error:', response.status, errorText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const caption = data.choices[0].message.content;

    console.log('Caption generated:', caption);

    return new Response(
      JSON.stringify({ caption }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );

  } catch (error) {
    console.error('Error in generate-caption:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});
