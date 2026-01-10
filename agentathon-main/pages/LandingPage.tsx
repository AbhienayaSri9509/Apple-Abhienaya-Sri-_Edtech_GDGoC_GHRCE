import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import { Camera, Type, MapPin, ArrowRight, Mic, MicOff } from 'lucide-react';

const LandingPage: React.FC = () => {
    const navigate = useNavigate();
    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState('');

    const options = [
        {
            title: 'Sign to Text',
            description: 'Translate sign language gestures into text in real-time with your camera.',
            icon: <Camera className="w-8 h-8 text-cyan-400" />,
            path: '/sign-to-text',
            color: 'hover:border-cyan-500/50 hover:shadow-cyan-500/20'
        },
        {
            title: 'Text to Sign',
            description: 'Type any text and watch our 3D avatar translate it into sign language.',
            icon: <Type className="w-8 h-8 text-purple-400" />,
            path: '/text-to-sign',
            color: 'hover:border-purple-500/50 hover:shadow-purple-500/20'
        },
        {
            title: 'Directions',
            description: 'Navigate complex environments with AI-assisted guidance.',
            icon: <MapPin className="w-8 h-8 text-pink-400" />,
            path: '/directions',
            color: 'hover:border-pink-500/50 hover:shadow-pink-500/20'
        }
    ];

    const startListening = () => {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert("Speech recognition is not supported in this browser.");
            setTranscript("Speech recognition not supported in this browser.");
            return;
        }

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        console.log("Starting speech recognition...");

        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            setIsListening(true);
            setTranscript("Listening...");
        };

        recognition.onresult = (event: any) => {
            const command = event.results[0][0].transcript.toLowerCase().trim();
            console.log("Speech recognition result:", command);
            setTranscript(command);
            
            // Small delay to ensure state is updated before navigation
            setTimeout(() => {
                handleCommand(command);
            }, 100);
            
            setIsListening(false);
        };

        recognition.onend = () => {
            console.log("Speech recognition ended.");
            setIsListening(false);
            // Don't clear transcript, so user can see the result/error
        };

        recognition.onerror = (event: any) => {
            console.error("Speech recognition error", event.error);
            setIsListening(false);
            if (event.error === 'no-speech') {
                setTranscript("No speech detected. Try again.");
            } else if (event.error === 'not-allowed') {
                setTranscript("Microphone permission denied. Please allow microphone access.");
            } else if (event.error === 'network') {
                setTranscript("Network error. Check your connection.");
            } else {
                setTranscript(`Error: ${event.error}. Try again.`);
            }
        };

        try {
            recognition.start();
        } catch (e) {
            console.error("Failed to start recognition", e);
            setIsListening(false);
        }
    };

    React.useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if ((event.code === 'Space' || event.key.toLowerCase() === 'm') && !isListening) {
                event.preventDefault(); // Prevent scrolling for Space
                startListening();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isListening]);

    const handleCommand = (command: string) => {
        if (!command) {
            console.log("Empty command received");
            return;
        }

        // Normalize command: lowercase and trim
        const normalizedCommand = command.toLowerCase().trim();
        console.log("Processing voice command:", normalizedCommand);

        // More flexible matching for "Sign to Text"
        if (normalizedCommand.includes('sign to text') || 
            normalizedCommand.includes('sign to test') || 
            normalizedCommand.includes('sign language') ||
            normalizedCommand.includes('sign language to text') ||
            (normalizedCommand.includes('camera') && normalizedCommand.includes('sign')) ||
            normalizedCommand === 'sign' ||
            normalizedCommand.includes('open sign')) {
            console.log("Navigating to /sign-to-text");
            setTranscript("Opening Sign to Text...");
            navigate('/sign-to-text');
            return;
        }

        // More flexible matching for "Text to Sign"
        if (normalizedCommand.includes('text to sign') || 
            normalizedCommand.includes('type to sign') ||
            normalizedCommand.includes('text to sign language') ||
            normalizedCommand.includes('avatar') ||
            normalizedCommand.includes('3d avatar') ||
            normalizedCommand === 'text' ||
            normalizedCommand.includes('open text')) {
            console.log("Navigating to /text-to-sign");
            setTranscript("Opening Text to Sign...");
            navigate('/text-to-sign');
            return;
        }

        // More flexible matching for "Directions"
        if (normalizedCommand.includes('direction') || 
            normalizedCommand.includes('directions') || 
            normalizedCommand.includes('navigate') || 
            normalizedCommand.includes('map') ||
            normalizedCommand.includes('navigation') ||
            normalizedCommand.includes('route') ||
            normalizedCommand.includes('guide') ||
            normalizedCommand === 'directions' ||
            normalizedCommand.includes('open direction')) {
            console.log("Navigating to /directions");
            setTranscript("Opening Directions...");
            navigate('/directions');
            return;
        }

        // If no match, show error
        console.log("Unknown command:", normalizedCommand);
        setTranscript(`Unknown: "${command}". Try saying "Sign to Text", "Text to Sign", or "Directions".`);
    };

    return (
        <Layout showBackButton={false}>
            <div className="flex flex-col items-center justify-center min-h-screen p-8 relative">
                {/* Hero Section */}
                <div className="text-center mb-16 space-y-4 max-w-2xl">
                    <h1 className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-200 via-indigo-200 to-purple-200">
                        Lumina
                    </h1>
                    <p className="text-xl text-slate-300 font-light tracking-wide">
                        Your Virtual Companion for Seamless Communication
                    </p>
                </div>

                {/* Options Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl">
                    {options.map((option) => (
                        <button
                            key={option.path}
                            onClick={() => navigate(option.path)}
                            className={`group relative flex flex-col items-start p-8 rounded-3xl bg-white/5 backdrop-blur-xl border border-white/10 transition-all duration-300 hover:-translate-y-2 hover:bg-white/10 ${option.color} cursor-pointer text-left`}
                        >
                            <div className="mb-6 p-4 rounded-2xl bg-white/5 group-hover:bg-white/10 transition-colors">
                                {option.icon}
                            </div>

                            <h2 className="text-2xl font-bold text-white mb-3 group-hover:text-white/90">
                                {option.title}
                            </h2>

                            <p className="text-slate-400 mb-8 flex-grow leading-relaxed">
                                {option.description}
                            </p>

                            <div className="flex items-center text-sm font-semibold uppercase tracking-wider text-white/50 group-hover:text-white transition-colors">
                                Launch
                                <ArrowRight className="ml-2 w-4 h-4 transition-transform group-hover:translate-x-1" />
                            </div>
                        </button>
                    ))}
                </div>

                {/* Voice Command Button */}
                <div className="fixed bottom-8 right-8 z-50 flex flex-col items-end gap-4">
                    {transcript && isListening && (
                        <div className="bg-black/80 text-white px-4 py-2 rounded-xl backdrop-blur-md border border-white/10">
                            {transcript}
                        </div>
                    )}
                    <button
                        onClick={startListening}
                        className={`p-4 rounded-full shadow-lg transition-all duration-300 ${isListening
                            ? 'bg-red-500 hover:bg-red-600 animate-pulse shadow-red-500/50'
                            : 'bg-cyan-500 hover:bg-cyan-600 shadow-cyan-500/50 hover:scale-110'
                            }`}
                        title="Voice Navigation"
                    >
                        {isListening ? (
                            <MicOff className="w-8 h-8 text-white" />
                        ) : (
                            <Mic className="w-8 h-8 text-white" />
                        )}
                    </button>
                </div>

                {/* Footer */}
                <div className="absolute bottom-8 text-slate-500 text-sm">
                    © 2025 Lumina Companion. Powered by Gemini & React Three Fiber.
                </div>
            </div>
        </Layout>
    );
};

export default LandingPage;
