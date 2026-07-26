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
    const { frames, mode } = await req.json();

    if (!frames || !Array.isArray(frames) || frames.length === 0) {
      throw new Error('No frames provided');
    }

    if (!mode || !['summary', 'timecapsule'].includes(mode)) {
      throw new Error('Invalid mode. Must be "summary" or "timecapsule"');
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    console.log(`Processing ${frames.length} frames in ${mode} mode...`);

    if (mode === 'summary') {
      const imageContents = frames.map((frame: string) => ({
        type: 'image_url',
        image_url: { url: frame }
      }));

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
                  text: 'These are key frames from a video shown in chronological order. Provide a concise summary (2-3 sentences) describing the main events, actions, and subjects. Ground every statement in what is visible across the frames: do not invent names, brands, sounds, dialogue, or events between frames; do not use hedging words ("appears", "seems", "possibly"). If the frames do not show something clearly, leave it out entirely.'
                },
                ...imageContents
              ]
            }
          ],
          max_tokens: 200,
          temperature: 0.2
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('AI gateway error:', response.status, errorText);
        throw new Error(`AI gateway error: ${response.status}`);
      }

      const data = await response.json();
      const summary = data.choices[0].message.content;

      console.log('Video summary generated:', summary);

      return new Response(
        JSON.stringify({ summary }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );

    } else {
      // Caption all frames concurrently — sequential requests made Time
      // Capsule mode take frames×latency; parallel takes ~1×latency.
      console.log(`Captioning ${frames.length} frames in parallel...`);

      const captions = await Promise.all(
        frames.map(async (frame: string, i: number) => {
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
                      text: 'Describe this video frame in one concise sentence. State only the main action, subject, and visible elements. Do not invent names, brands, or context outside the frame, and do not hedge with "appears" or "seems".'
                    },
                    {
                      type: 'image_url',
                      image_url: { url: frame }
                    }
                  ]
                }
              ],
              max_tokens: 100,
              temperature: 0.2
            }),
          });

          if (!response.ok) {
            const errorText = await response.text();
            console.error(`AI gateway error on frame ${i + 1}:`, response.status, errorText);
            throw new Error(`AI gateway error: ${response.status}`);
          }

          const data = await response.json();
          return data.choices[0].message.content as string;
        })
      );

      console.log('All frame captions generated');

      return new Response(
        JSON.stringify({ captions }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

  } catch (error) {
    console.error('Error in analyze-video:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});
