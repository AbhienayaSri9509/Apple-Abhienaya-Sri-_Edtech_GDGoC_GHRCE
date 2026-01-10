import { SignJWT, importPKCS8 } from 'jose';
import credentials from './service-account.json';

type VoiceConfig = {
  languageCode?: string;
  name?: string;
  ssmlGender?: 'MALE' | 'FEMALE' | 'NEUTRAL';
};

let cachedToken: string | null = null;
let tokenExpiry: number | null = null;

async function getAccessToken(): Promise<string> {
  // Return cached token if still valid (minus 1 min buffer)
  if (cachedToken && tokenExpiry && Date.now() < tokenExpiry) {
    return cachedToken;
  }

  try {
    const { client_email, private_key, token_uri } = credentials;

    // Import the private key
    // The private key in the JSON includes \n, which jose handles correctly
    const privateKey = await importPKCS8(private_key, 'RS256');

    // Create a signed JWT for OAuth2
    const jwt = await new SignJWT({
      scope: 'https://www.googleapis.com/auth/cloud-platform'
    })
      .setProtectedHeader({ alg: 'RS256' })
      .setIssuer(client_email)
      .setAudience(token_uri)
      .setExpirationTime('1h')
      .setIssuedAt()
      .sign(privateKey);

    // Exchange JWT for Access Token
    const response = await fetch(token_uri, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        assertion: jwt,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Token fetch error:', errorText);
      throw new Error(`Failed to authenticate with Service Account: ${response.status}`);
    }

    const data = await response.json();
    cachedToken = data.access_token;
    // Set expiry based on response (usually 3600s), subtract buffer
    tokenExpiry = Date.now() + (data.expires_in * 1000) - 30000;

    return cachedToken!;
  } catch (error) {
    console.error('Error getting access token:', error);
    throw error;
  }
}

export async function synthesizeSpeech(
  text: string,
  {
    languageCode = 'en-US',
    ssmlGender = 'FEMALE',
    name,
  }: VoiceConfig = {}
): Promise<string> {
  try {
    const accessToken = await getAccessToken();

    const endpoint = `https://texttospeech.googleapis.com/v1/text:synthesize`;

    const body = {
      input: { text },
      voice: {
        languageCode,
        ssmlGender,
        ...(name ? { name } : {}),
      },
      audioConfig: {
        audioEncoding: 'MP3',
        speakingRate: 1.0,
        pitch: 0.0,
      },
    };

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`, // Use Bearer token
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => '');
      throw new Error(`TTS request failed (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const audioContent = data?.audioContent;
    if (!audioContent) {
      throw new Error('No audio content returned from TTS.');
    }

    // Return data URL for quick playback
    return `data:audio/mp3;base64,${audioContent}`;
  } catch (err) {
    console.error('TTS Synthesis Error:', err);
    throw err;
  }
}

