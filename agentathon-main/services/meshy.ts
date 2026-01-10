// Meshy Text-to-3D helper for the Text to Sign page
// Handles task creation and status polling

export interface MeshyTaskResult {
  status?: string;
  message?: string;
  result?: {
    preview_image?: string;
    preview_image_url?: string;
    thumbnail_url?: string;
    model_url?: string;
    model_urls?: Record<string, string>;
    [key: string]: any;
  };
  [key: string]: any;
}

const API_BASE = 'https://api.meshy.ai/openapi/v2';

const resolveMeshyKey = (): string => {
  const key =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_MESHY_API_KEY) ||
    (typeof process !== 'undefined' ? process.env?.MESHY_API_KEY : undefined);

  if (!key) {
    throw new Error('Missing Meshy API key. Add VITE_MESHY_API_KEY to your environment.');
  }

  return key;
};

export const requestTextToSignTask = async (userText: string): Promise<{ taskId: string; prompt: string }> => {
  const apiKey = resolveMeshyKey();
  const prompt = `Generate a realistic sign language pose of a man expressing: "${userText}". Focus on clear hand shapes and body positioning from a frontal view.`;

  const payload = {
    mode: 'preview',
    prompt,
    negative_prompt: 'low quality, low resolution, low poly, ugly',
    art_style: 'realistic',
    should_remesh: true,
  };

  const response = await fetch(`${API_BASE}/text-to-3d`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Meshy task creation failed (${response.status}): ${errorText || response.statusText}`);
  }

  const data = await response.json();
  const taskId = data?.result;

  if (!taskId) {
    throw new Error('Meshy did not return a task id.');
  }

  return { taskId, prompt };
};

export const fetchTextToSignTask = async (taskId: string): Promise<MeshyTaskResult> => {
  const apiKey = resolveMeshyKey();

  const response = await fetch(`${API_BASE}/text-to-3d/${taskId}`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Meshy status check failed (${response.status}): ${errorText || response.statusText}`);
  }

  return response.json();
};

