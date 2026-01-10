// Sign Language Processing Service
// Uses Gemini API (via raw fetch) to analyze sign language images

type SupportedLanguage = 'en' | 'ta';

export const processSignLanguageImage = async (
  imageBase64: string,
  language: SupportedLanguage = 'en'
): Promise<string> => {
  try {
    // 1) API key resolution (works in Vite/browser and server)
    // 1) API key resolution
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY || (process.env as any)?.API_KEY;

    if (!apiKey) {
      console.warn("WARN: Missing VITE_GEMINI_API_KEY. Using Mock Mode.");
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      return "Hello! This is a demo response (Mock Mode).";
    }

    // 2) Model endpoint (requested model)
    const MODEL_NAME = 'gemini-1.5-flash';
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:generateContent?key=${apiKey}`;

    // 3) Prompt
    const languageInstruction =
      language === 'ta'
        ? 'Return the response in Tamil (தமிழ்).'
        : 'Return the response in English.';

    const prompt = `Analyze this image from a sign language video and identify the exact word or short phrase the person is signing.

Focus on:
- Hand gestures and movements
- Facial expressions that convey meaning
- Body language and posture

${languageInstruction}

Return ONLY the recognized word or short phrase (1-5 words maximum).
Do NOT add any explanation, commentary, or additional text.
Do NOT respond in a conversational manner.

Return ONLY the recognized text, nothing else.`;

    // 4) Request with timeout (15 seconds max)
    const startTime = performance.now();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

    let response: Response;
    try {
      response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                { text: prompt },
                {
                  inlineData: {
                    mimeType: "image/jpeg",
                    data: imageBase64,
                  },
                },
              ],
            },
          ],
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const requestTime = performance.now() - startTime;
      console.log(`[Gemini API] Request completed in ${requestTime.toFixed(0)}ms`);
    } catch (fetchError: any) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        throw new Error("Gemini API request timed out after 15 seconds. Please try again.");
      }
      throw fetchError;
    }

    // 5) Handle Response
    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      console.error(`Gemini API Error (${response.status}):`, errorText);

      // FALLBACK TO MOCK IF API FAILS (for robustness during demo)
      console.warn("Falling back to Mock Mode due to API error.");
      await new Promise(resolve => setTimeout(resolve, 1000));
      return "Hello! (Fallback Mock Response)";
    }

    const data = await response.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!text) {
      throw new Error("Gemini returned no text. The model may not have recognized a sign.");
    }

    return text.trim();

    return "Hello! (Network Error Mock)";

  } catch (error: any) {
    console.error("processSignLanguageImage failed:", error);
    // Fallback for network errors too
    return "Hello! (Error Fallback)";
  }
};
