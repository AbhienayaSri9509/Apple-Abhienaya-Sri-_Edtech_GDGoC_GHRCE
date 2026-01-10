// import credentials from './service-account.json';
// NOTE: Google Cloud Service Account JSON cannot be imported in client-side code securely 
// and has been removed from the repository. All TTS now defaults to Browser Native API.

// If you want to use Google Cloud TTS, you would need a backend proxy to handle auth.
// For this demo, we will use window.speechSynthesis (Web Speech API).

type VoiceConfig = {
  languageCode?: string;
  name?: string;
  ssmlGender?: 'MALE' | 'FEMALE' | 'NEUTRAL';
};

async function getAccessToken(): Promise<string> {
  // Placeholder for backend token retrieval if implemented later
  return "";
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
    // FALLBACK: Use Web Speech API (Browser Native)
    // Since we cannot safely expose Google Cloud keys in a client-side Vite app,
    // we use the browser's built-in TTS which is free and requires no keys.

    console.log(`TTS (Native): Speaking "${text}"`);

    // Cancel any current speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    // utterance.lang = languageCode; // e.g. 'en-US'
    // utterance.rate = 1.0;

    // Simple basic voice selection if possible
    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find(v => v.name.includes('Female') || v.name.includes('Google')) || voices[0];
    if (voice) utterance.voice = voice;

    // We wrap this in a promise to simulate the async nature, 
    // but the caller expects a URL to *play*. 
    // Since we can't get a URL from speechSyn, we play it HERE, 
    // and return a silent audio URL to satisfy the caller's `<audio>` requirement.

    window.speechSynthesis.speak(utterance);

    // 1-second of silence MP3 (base64) to prevent errors in caller
    const silentMp3 = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTSVMAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAw//OEAAAAAAAAAAAAAAAAAAAAAAAATGF2YzU4LjU0LjEwMAAAAAAAAAAAAAAAI1RTSVMAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAw";

    return Promise.resolve(silentMp3);

  } catch (err) {
    console.error('TTS Synthesis Error:', err);
    throw err;
  }
}

