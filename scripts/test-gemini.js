// Simple connectivity test for Gemini
// Usage: GEMINI_API_KEY=your_key_here npm run test:gemini

import { GoogleGenerativeAI } from '@google/generative-ai';

async function main() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error('GEMINI_API_KEY is not set. Add it to your environment or .env.local/.env.');
    process.exit(1);
  }

  try {
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = "Reply with exactly: API is working";
    const result = await model.generateContent(prompt);
    const text = await result.response.text();

    console.log('Gemini test response:', text.trim());
    if (text.trim() === 'API is working') {
      console.log('✅ Gemini API is working');
      process.exit(0);
    } else {
      console.error('⚠️ Unexpected response. Check model/permissions:', text.trim());
      process.exit(2);
    }
  } catch (err) {
    console.error('❌ Gemini test failed:', err?.message || err);
    process.exit(1);
  }
}

main();

