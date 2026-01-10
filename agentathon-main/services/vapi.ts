type VapiResponse = {
  // Optional speech audio payload (base64 mp3 or wav)
  outputAudio?: string;
  // Optional plain text response
  text?: string;
  // Optional destination extracted by the agent
  destination?: string;
  // Optional generic payload the agent might return
  data?: Record<string, unknown>;
  args?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
};

const VAPI_URL =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_VAPI_API_URL) ||
  (typeof process !== 'undefined' && process.env?.VITE_VAPI_API_URL) ||
  // Default to the provided endpoint if no env is set
  'https://api.yourservice.com';

const VAPI_API_KEY =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_VAPI_API_KEY) ||
  (typeof process !== 'undefined' && process.env?.VITE_VAPI_API_KEY) ||
  '';

/**
 * Sends the transcribed user text to the Vapi voice agent HTTP endpoint.
 * Expects the backend to return JSON and optionally an audio payload.
 */
export async function askVapi(text: string): Promise<VapiResponse> {
  if (!VAPI_URL || !VAPI_API_KEY) {
    throw new Error('Missing VAPI config. Set VITE_VAPI_API_URL and VITE_VAPI_API_KEY.');
  }

  const response = await fetch(VAPI_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${VAPI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    throw new Error(`Vapi request failed: ${response.status} ${errText}`);
  }

  // Try to parse JSON; if it fails, throw
  const data = await response.json();
  return data as VapiResponse;
}

