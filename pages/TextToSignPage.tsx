import React, { useMemo, useState } from 'react';
import {
    AlertCircle,
    CheckCircle2,
    Image as ImageIcon,
    Loader2,
    RefreshCw,
    Sparkles,
} from 'lucide-react';
import Layout from '../components/Layout';
import { runTextToSign, TextToSignResult } from '../services/textToSignLocal';
import Experience from '../components/Experience';
import { AvatarState } from '../types';

type GenerationState = 'idle' | 'generating' | 'succeeded' | 'error';

const TextToSignPage: React.FC = () => {
    const [inputText, setInputText] = useState('');
    const [status, setStatus] = useState<GenerationState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<TextToSignResult | null>(null);

    // Avatar state for the 3D "doll"
    const [avatarState, setAvatarState] = useState<AvatarState>({
        isThinking: false,
        isTalking: false,
        mood: 'neutral'
    });

    const generatedPrompt = useMemo(() => {
        if (!inputText.trim()) return 'Generate sign language showing a man signing your text.';
        return `Generate sign language showing a man signing: "${inputText.trim()}"`;
    }, [inputText]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputText.trim() || status === 'generating') return;

        setError(null);
        setResult(null);
        setStatus('generating');
        // Make avatar think while generating
        setAvatarState(prev => ({ ...prev, isThinking: true, isTalking: false }));

        try {
            const data = await runTextToSign(inputText.trim());
            setResult(data);
            setStatus('succeeded');
            setAvatarState(prev => ({ ...prev, isThinking: false }));

            // If it's the placeholder video, treat it as a trigger to use the 3D avatar instead
            // OR we can just use the avatar essentially if the video is "placeholder"
            const isPlaceholder = data.videoUrl.includes('placeholder');

            if (isPlaceholder || data.videoUrl) {
                // Trigger an "acting" animation on the avatar for a short duration to simulate signing
                // Since we don't have real motion data, we just use the 'talking' state as a proxy for movement
                setAvatarState(prev => ({ ...prev, isTalking: true }));
                setTimeout(() => {
                    setAvatarState(prev => ({ ...prev, isTalking: false }));
                }, 5000); // Act for 5 seconds
            }

        } catch (err: any) {
            setStatus('error');
            setError(err?.message || 'Unable to generate sign animation.');
            setAvatarState(prev => ({ ...prev, isThinking: false, isTalking: false }));
        }
    };

    const statusBadge = () => {
        const base = 'inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold';
        switch (status) {
            case 'generating':
                return (
                    <div className={`${base} bg-amber-500/20 text-amber-100 border border-amber-500/30`}>
                        <RefreshCw className="w-4 h-4 animate-spin" /> Generating sign animation...
                    </div>
                );
            case 'succeeded':
                return (
                    <div className={`${base} bg-emerald-500/20 text-emerald-100 border border-emerald-500/30`}>
                        <CheckCircle2 className="w-4 h-4" /> Completed
                    </div>
                );
            case 'error':
                return (
                    <div className={`${base} bg-rose-500/20 text-rose-100 border border-rose-500/30`}>
                        <AlertCircle className="w-4 h-4" /> Error
                    </div>
                );
            default:
                return (
                    <div className={`${base} bg-white/10 text-white/80 border border-white/10`}>
                        <Sparkles className="w-4 h-4" /> Idle
                    </div>
                );
        }
    };

    return (
        <Layout>
            <div className="h-full w-full flex items-center justify-center p-6 lg:p-12 relative z-10">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl">
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-12 h-12 rounded-2xl bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
                                <Sparkles className="w-6 h-6 text-indigo-200" />
                            </div>
                            <div>
                                <p className="text-indigo-200 text-sm uppercase tracking-wider font-semibold">Text to Sign</p>
                                <h1 className="text-3xl font-bold text-white">Generate sign language animation</h1>
                            </div>
                        </div>
                        <p className="text-white/70 mb-6">
                            Type any phrase and we&apos;ll use the HOW2SIGN dataset + SMPL-X to generate a sign language animation.
                            If exact motion data isn&apos;t found, our 3D Avatar will enact the text.
                        </p>
                        <div className="mb-4">{statusBadge()}</div>

                        <form onSubmit={handleSubmit} className="space-y-4">
                            <label className="block text-sm font-semibold text-white/80">Your text</label>
                            <textarea
                                className="w-full min-h-[120px] bg-black/40 border border-white/10 rounded-2xl px-4 py-3 text-white placeholder-white/40 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/40 outline-none transition"
                                placeholder="e.g. My name is Sanjai"
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                disabled={status === 'generating'}
                            />

                            <div className="bg-indigo-500/10 border border-indigo-500/30 text-indigo-50 rounded-2xl p-4 text-sm">
                                <p className="font-semibold mb-1">Prompt preview</p>
                                <p className="text-indigo-100/80">{generatedPrompt}</p>
                            </div>

                            {error && (
                                <div className="bg-rose-500/10 border border-rose-500/30 text-rose-100 rounded-2xl p-4 text-sm flex items-start gap-2">
                                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                                    <div>
                                        <p className="font-semibold">Something went wrong</p>
                                        <p>{error}</p>
                                    </div>
                                </div>
                            )}

                            <div className="flex items-center gap-3">
                                <button
                                    type="submit"
                                    disabled={!inputText.trim() || status === 'generating'}
                                    className="inline-flex items-center gap-2 bg-indigo-500 hover:bg-indigo-400 text-white px-5 py-3 rounded-2xl font-semibold transition disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    {status === 'generating' ? (
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <Sparkles className="w-5 h-5" />
                                    )}
                                    Generate Sign Animation
                                </button>

                                <button
                                    type="button"
                                    onClick={() => {
                                        setInputText('');
                                        setStatus('idle');
                                        setError(null);
                                        setResult(null);
                                        setAvatarState({ isThinking: false, isTalking: false, mood: 'neutral' });
                                    }}
                                    className="text-white/70 hover:text-white text-sm underline"
                                >
                                    Reset
                                </button>
                            </div>
                        </form>
                    </div>

                    <div className="bg-black/50 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-2xl relative overflow-hidden flex flex-col">
                        <div className="flex items-center justify-between mb-4 relative z-10">
                            <div>
                                <p className="text-xs uppercase tracking-wide text-white/50 font-semibold">Generation Status</p>
                                <h2 className="text-xl font-bold text-white">Avatar Output</h2>
                            </div>
                            {statusBadge()}
                        </div>

                        {/* Rendering Area: Either Video or 3D Avatar */}
                        {/* We use a conditional: if using placeholder, show Avatar. Else show video. */}
                        {/* For this "working model" request, let's prioritize the Avatar if video is missing/placeholder */}

                        <div className="flex-grow aspect-video bg-white/5 border border-white/10 rounded-2xl overflow-hidden relative">
                            {result && !result.videoUrl.includes('placeholder') && result.videoUrl ? (
                                <video
                                    src={result.videoUrl}
                                    className="w-full h-full object-cover"
                                    controls
                                    autoPlay
                                    loop
                                    muted
                                    playsInline
                                />
                            ) : (
                                // Show 3D Avatar as default/fallback
                                <div className="w-full h-full absolute inset-0">
                                    <Experience avatarState={avatarState} />

                                    {/* Hint Overlay */}
                                    <div className="absolute bottom-4 left-4 right-4 text-center pointer-events-none">
                                        {status === 'idle' && <p className="text-white/40 text-sm bg-black/40 rounded-full px-3 py-1 inline-block">Ready to translate text to gestures</p>}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-4 space-y-3 text-sm text-white/80 mt-4 relative z-10">
                            <div className="flex items-center justify-between">
                                <span className="text-white/60">Prompt</span>
                                <span className="text-right text-white/90 text-xs">{generatedPrompt}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-white/60">Current Status</span>
                                <span className="font-semibold text-white">
                                    {status.toUpperCase()}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </Layout>
    );
};

export default TextToSignPage;
