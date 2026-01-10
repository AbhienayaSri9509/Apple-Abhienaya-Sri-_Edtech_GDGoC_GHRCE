import React, { useState, useEffect, useCallback } from 'react';
import Experience from '../components/Experience';
import ChatInterface from '../components/ChatInterface';
import SignLanguageCamera from '../components/SignLanguageCamera';
import SignLanguageResponse from '../components/SignLanguageResponse';
import { sendMessageToGemini, initializeChat } from '../services/gemini';
import { synthesizeSpeech } from '../services/tts';
import { Message, ChatStatus, AvatarState } from '../types';
import Layout from '../components/Layout';

const SignToTextPage: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [status, setStatus] = useState<ChatStatus>(ChatStatus.IDLE);
    const [detectedSignText, setDetectedSignText] = useState<string>('');
    const [avatarState, setAvatarState] = useState<AvatarState>({
        isThinking: false,
        isTalking: false,
        mood: 'neutral'
    });

    // Initialize chat on mount
    useEffect(() => {
        initializeChat();
    }, []);

    const handleSignDetected = useCallback((text: string) => {
        setDetectedSignText(text);
        // Kick off talking animation immediately after sign response
        setAvatarState(prev => ({ ...prev, isTalking: true }));
    }, []);

    const playSignAudio = useCallback(async (text: string) => {
        try {
            setAvatarState(prev => ({ ...prev, isTalking: true }));
            // Use English voice by default; adjust languageCode if you add localization.
            const audioUrl = await synthesizeSpeech(text, { languageCode: 'en-US', ssmlGender: 'FEMALE' });
            const audio = new Audio(audioUrl);
            audio.play().catch(err => console.error('Audio play error:', err));
            audio.onended = () => {
                setAvatarState(prev => ({ ...prev, isTalking: false }));
            };
        } catch (err) {
            console.error('TTS error:', err);
            setAvatarState(prev => ({ ...prev, isTalking: false }));
        }
    }, []);

    const handleSignProcessing = useCallback((isProcessing: boolean) => {
        setAvatarState(prev => ({
            ...prev,
            isThinking: isProcessing
        }));
    }, []);

    const handleSendMessage = useCallback(async (text: string) => {
        // Add user message immediately
        const userMessage: Message = {
            role: 'user',
            content: text,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);
        setStatus(ChatStatus.LOADING);
        setAvatarState(prev => ({ ...prev, isThinking: true }));

        try {
            // API Call
            const responseText = await sendMessageToGemini(text);

            const botMessage: Message = {
                role: 'model',
                content: responseText,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, botMessage]);
            setStatus(ChatStatus.IDLE);

            try {
                setAvatarState(prev => ({ ...prev, isThinking: false, isTalking: true }));
                const audioUrl = await synthesizeSpeech(responseText, { languageCode: 'en-US', ssmlGender: 'FEMALE' });
                const audio = new Audio(audioUrl);

                audio.onended = () => {
                    setAvatarState(prev => ({ ...prev, isTalking: false }));
                };

                await audio.play();
            } catch (ttsError) {
                console.error('TTS playback error:', ttsError);
                // Fallback or just stop talking animation
                setAvatarState(prev => ({ ...prev, isTalking: false }));
            }

        } catch (error) {
            console.error(error);
            setStatus(ChatStatus.ERROR);
            setAvatarState(prev => ({ ...prev, isThinking: false }));

            const errorMessage: Message = {
                role: 'model',
                content: "I'm having trouble connecting right now. Can we try again?",
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        }
    }, []);

    return (
        <Layout>
            {/* 3D Scene */}
            <div className="absolute inset-0 z-0">
                <Experience avatarState={avatarState} />
            </div>

            {/* UI Overlay */}
            <ChatInterface
                messages={messages}
                status={status}
                onSendMessage={handleSendMessage}
            />

            {/* Sign Language Camera - Left Side */}
            <SignLanguageCamera
                onSignDetected={(text) => {
                    handleSignDetected(text);
                    playSignAudio(text);
                }}
                onProcessingChange={handleSignProcessing}
            />

            {/* Sign Language Response - Right Side */}
            <SignLanguageResponse detectedText={detectedSignText} />
        </Layout>
    );
};

export default SignToTextPage;
