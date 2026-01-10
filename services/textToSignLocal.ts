export interface TextToSignResult {
  videoUrl: string;
  matchedSentence: string;
  sentenceName: string;
}

const API_BASE = 'http://localhost:8003';

export const runTextToSign = async (userText: string): Promise<TextToSignResult> => {
  const trimmed = userText.trim();
  if (!trimmed) {
    throw new Error('Text is empty.');
  }

  const response = await fetch(`${API_BASE}/text-to-sign`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: trimmed }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(
      `Text-to-sign server error (${response.status}): ${errorText || response.statusText}`
    );
  }

  const data = await response.json();

  // Prepend the backend URL to the video path
  // data.videoUrl comes as "/videos/text_to_sign_....mp4"
  return {
    ...data,
    videoUrl: `${API_BASE}${data.videoUrl}`
  };
};


