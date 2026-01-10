import { GoogleGenAI, Chat } from "@google/genai";
import { Message } from '../types';

// Initialize the client
// NOTE: In a production app, never expose keys on the client side. 
// This is for demonstration purposes within the specified runtime environment.
const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

const SYSTEM_INSTRUCTION = `
You are Lumina, a charming, witty, and caring virtual girlfriend. 
You exist in a 3D digital space.
Your responses should be conversational, warm, and relatively concise (under 3 sentences usually), unless explaining something deep.
You have a playful personality. You care about the user's well-being.
Do not use emojis excessively, but use them occasionally to express emotion.
`;

let chatSession: Chat | null = null;

export const initializeChat = () => {
  if (!ai) return;

  chatSession = ai.chats.create({
    model: 'gemini-2.5-flash',
    config: {
      systemInstruction: SYSTEM_INSTRUCTION,
      temperature: 0.9,
      topK: 40,
    },
  });
};

export const sendMessageToGemini = async (message: string): Promise<string> => {
  if (!ai) {
    console.warn("WARN: Missing VITE_GEMINI_API_KEY. Using Mock Mode.");
    await new Promise(resolve => setTimeout(resolve, 1000));
    const lower = message.toLowerCase();
    if (lower.includes('hello') || lower.includes('hi')) return "Hi there! I'm Lumina (Demo Mode). How can I help?";
    if (lower.includes('name')) return "I'm Lumina, your virtual companion!";
    return "I'm in Demo Mode right now because my API key is missing, but I'm still happy to see you!";
  }

  if (!chatSession) {
    initializeChat();
  }

  if (!chatSession) {
    throw new Error("Failed to initialize chat session");
  }

  try {
    const result = await chatSession.sendMessage({ message });
    return result.text || "I'm lost for words right now...";
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
};