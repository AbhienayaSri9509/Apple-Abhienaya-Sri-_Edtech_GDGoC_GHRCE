import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, Navigation, AlertCircle, Loader2 } from 'lucide-react';
import { askVapi } from '../services/vapi';

// Let TypeScript know the Google Maps script will attach a global
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const google: any;

// Access environment variables - these will be replaced at build time
const VITE_GOOGLE_MAPS_API_KEY =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_GOOGLE_MAPS_API_KEY) ||
    (typeof process !== 'undefined' && process.env?.VITE_GOOGLE_MAPS_API_KEY) ||
    '';

const VITE_GOOGLE_CLOUD_TTS_API_KEY =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_GOOGLE_CLOUD_TTS_API_KEY) ||
    (typeof process !== 'undefined' && process.env?.VITE_GOOGLE_CLOUD_TTS_API_KEY) ||
    '';

interface NavState {
    step: 'idle' | 'listening' | 'processing' | 'ready' | 'navigating' | 'error';
    destination: string;
    error?: string;
}

const mapContainerStyle = {
    width: '100%',
    height: '100%',
};

const defaultCenter = {
    lat: 40.7128,
    lng: -74.0060,
};

// Announce next step when within this distance (meters)
const STEP_ANNOUNCEMENT_DISTANCE = 50; // 50 meters before the turn
const STEP_COMPLETION_DISTANCE = 15; // Move to next step when within 15m

const BlindNavigation: React.FC = () => {
    // State
    const [navState, setNavState] = useState<NavState>({ step: 'idle', destination: '' });
    const [currentLocation, setCurrentLocation] = useState<{ lat: number, lng: number } | null>(null);
    const [locationReady, setLocationReady] = useState<boolean>(false);
    const [directionsResponse, setDirectionsResponse] = useState<any>(null);
    const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
    const [instruction, setInstruction] = useState<string>("Tap screen to begin navigation");
    const [lastAnnouncedStep, setLastAnnouncedStep] = useState<number>(-1);
    const [map, setMap] = useState<any>(null);

    // Refs
    const recognitionRef = useRef<any>(null);
    const watchIdRef = useRef<number | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const directionsServiceRef = useRef<any>(null);
    const directionsRendererRef = useRef<any>(null);
    const listenRetryRef = useRef<number>(0);

    // Load Google Maps Script
    // Load Google Maps Script
    useEffect(() => {
        if (!VITE_GOOGLE_MAPS_API_KEY) {
            // Demo mode - no map script
            console.warn("Using BlindNavigation in Demo Mode (No API Key)");
            return;
        }

        // Check if script already exists
        const existingScript = document.querySelector('script[src*="maps.googleapis.com/maps/api/js"]');
        if (existingScript) return;

        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${VITE_GOOGLE_MAPS_API_KEY}&libraries=places`;
        script.async = true;
        script.defer = true;
        document.head.appendChild(script);

        return () => {
            // Optional: don't remove if other components need it, or remove if you want strict scoping
            // document.head.removeChild(script);
        };
    }, []);

    // Initialize Map
    useEffect(() => {
        const initMap = () => {
            if (typeof google === 'undefined') {
                setTimeout(initMap, 100);
                return;
            }

            const mapElement = document.getElementById('map');
            if (!mapElement) return;

            const googleMap = new google.maps.Map(mapElement, {
                center: currentLocation || defaultCenter,
                zoom: 18,
                disableDefaultUI: true,
                styles: [
                    { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
                    { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
                    { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
                ]
            });

            setMap(googleMap);
            directionsServiceRef.current = new google.maps.DirectionsService();
            directionsRendererRef.current = new google.maps.DirectionsRenderer({
                map: googleMap,
                polylineOptions: { strokeColor: "#4f46e5", strokeWeight: 6 }
            });
        };

        initMap();
    }, []);

    // Text-to-Speech Helper
    const speak = useCallback(async (text: string, onEnd?: () => void) => {
        try {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current.currentTime = 0;
                audioRef.current = null;
            }

            if (!VITE_GOOGLE_CLOUD_TTS_API_KEY) {
                throw new Error("Missing VITE_GOOGLE_CLOUD_TTS_API_KEY env");
            }

            const response = await fetch(`https://texttospeech.googleapis.com/v1/text:synthesize?key=${VITE_GOOGLE_CLOUD_TTS_API_KEY}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: { text },
                    voice: { languageCode: 'en-US', ssmlGender: 'FEMALE' },
                    audioConfig: { audioEncoding: 'MP3' }
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`TTS request failed: ${response.status} ${errText}`);
            }

            const data = await response.json();
            if (!data?.audioContent) {
                throw new Error("No audioContent returned from TTS");
            }

            const audioUrl = `data:audio/mp3;base64,${data.audioContent}`;
            const audio = new Audio(audioUrl);
            audioRef.current = audio;
            if (onEnd) audio.onended = onEnd;
            await audio.play();
        } catch (error) {
            console.error("TTS Error:", error);
            // If TTS fails (e.g., missing key or network), still proceed to next action
            if (onEnd) onEnd();
        }
    }, []);

    // Calculate distance between two points (Haversine formula)
    const calculateDistance = useCallback((lat1: number, lng1: number, lat2: number, lng2: number): number => {
        const R = 6371e3; // Earth radius in meters
        const φ1 = lat1 * Math.PI / 180;
        const φ2 = lat2 * Math.PI / 180;
        const Δφ = (lat2 - lat1) * Math.PI / 180;
        const Δλ = (lng2 - lng1) * Math.PI / 180;

        const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }, []);

    // Start Location Tracking
    const startTracking = useCallback(() => {
        if (!navigator.geolocation) {
            speak("Geolocation is not supported on this device.");
            return;
        }

        const geoOptions = { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 };

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const newPos = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude,
                };
                setCurrentLocation(newPos);
                setLocationReady(true);
                if (map) map.panTo(newPos);
            },
            (err) => {
                console.error("Initial location error:", err);
            },
            geoOptions
        );

        watchIdRef.current = navigator.geolocation.watchPosition(
            (position) => {
                const newPos = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude,
                };
                setCurrentLocation(newPos);
                setLocationReady(true);
                if (map) map.panTo(newPos);
            },
            (err) => {
                console.error("Tracking Error:", err);
                if (err.code === 1) {
                    speak("Please allow location access to use navigation.");
                }
            },
            geoOptions
        );
    }, [speak, map]);

    useEffect(() => {
        if (map) {
            startTracking();
        }
        return () => {
            if (watchIdRef.current !== null) {
                navigator.geolocation.clearWatch(watchIdRef.current);
            }
        };
    }, [map, startTracking]);

    // Navigation Step Monitoring
    useEffect(() => {
        if (navState.step !== 'navigating' || !currentLocation || !directionsResponse) return;

        const route = directionsResponse.routes[0];
        const legs = route.legs[0];
        const steps = legs.steps;

        if (currentStepIndex >= steps.length) {
            speak("You have arrived at your destination!");
            setNavState({ step: 'idle', destination: '' });
            setInstruction("Journey complete!");
            return;
        }

        const currentStep = steps[currentStepIndex];
        const stepEndLat = currentStep.end_location.lat();
        const stepEndLng = currentStep.end_location.lng();

        const distanceToStepEnd = calculateDistance(
            currentLocation.lat,
            currentLocation.lng,
            stepEndLat,
            stepEndLng
        );

        // Announce step if within announcement distance and not yet announced
        if (distanceToStepEnd <= STEP_ANNOUNCEMENT_DISTANCE && lastAnnouncedStep !== currentStepIndex) {
            const rawInstruction = currentStep.instructions;
            const cleanInstruction = rawInstruction.replace(/<[^>]*>/g, '');
            const distance = currentStep.distance?.text || '';

            const announcement = `In ${distance}, ${cleanInstruction}`;
            setInstruction(cleanInstruction);
            speak(announcement);
            setLastAnnouncedStep(currentStepIndex);
        }

        // Move to next step when close enough to current step end
        if (distanceToStepEnd < STEP_COMPLETION_DISTANCE) {
            const nextIndex = currentStepIndex + 1;
            if (nextIndex < steps.length) {
                setCurrentStepIndex(nextIndex);
            } else {
                speak("You have arrived at your destination!");
                setNavState({ step: 'idle', destination: '' });
                setInstruction("Journey complete!");
            }
        }
    }, [currentLocation, navState.step, directionsResponse, currentStepIndex, lastAnnouncedStep, speak, calculateDistance]);

    // Calculate Route
    const calculateRoute = useCallback(async (destination: string) => {
        console.log("Calculate route called with:", destination);
        console.log("Current location:", currentLocation);
        console.log("Location ready:", locationReady);

        if (!currentLocation || !locationReady) {
            speak("I still don't have your location. Please enable GPS and try again.");
            setNavState(prev => ({ ...prev, step: 'error', destination, error: "Location unavailable" }));
            return;
        }

        if (!directionsServiceRef.current) {
            speak("Maps service not ready. Please wait a moment and try again.");
            setNavState(prev => ({ ...prev, step: 'error', destination, error: "Maps not ready" }));
            return;
        }

        if (typeof google === 'undefined') {
            speak("Google Maps is still loading. Please wait a moment.");
            setNavState(prev => ({ ...prev, step: 'error', destination, error: "Maps loading" }));
            return;
        }

        try {
            // Use geocoder first to validate destination
            const geocoder = new google.maps.Geocoder();
            const geocodeResult = await new Promise<any>((resolve, reject) => {
                geocoder.geocode({ address: destination }, (results: any, status: any) => {
                    if (status === 'OK' && results && results[0]) {
                        resolve(results[0].formatted_address);
                    } else {
                        reject(`Geocode failed: ${status}`);
                    }
                });
            });

            console.log("Geocoded destination:", geocodeResult);

            // Now get directions
            const results = await new Promise((resolve, reject) => {
                const request = {
                    origin: new google.maps.LatLng(currentLocation.lat, currentLocation.lng),
                    destination: geocodeResult,
                    travelMode: google.maps.TravelMode.WALKING,
                };

                console.log("Directions request:", request);

                directionsServiceRef.current.route(
                    request,
                    (result: any, status: any) => {
                        console.log("Directions status:", status);
                        console.log("Directions result:", result);

                        if (status === 'OK' && result) {
                            resolve(result);
                        } else {
                            reject(`Directions failed: ${status}`);
                        }
                    }
                );
            });

            setDirectionsResponse(results);
            if (directionsRendererRef.current) {
                directionsRendererRef.current.setDirections(results);
            }

            setNavState({ step: 'ready', destination });
            setCurrentStepIndex(0);
            setLastAnnouncedStep(-1);

            const leg = (results as any).routes[0].legs[0];
            const routeInfo = `Route found. Distance is ${leg.distance?.text}. Duration approximately ${leg.duration?.text}. Tap screen or say start to begin navigation.`;
            setInstruction(routeInfo);
            speak(routeInfo);

        } catch (error: any) {
            console.error("Route Error:", error);
            let errorMsg = "Route not found";
            let speakMsg = "I could not find a walking route to that location. Please try a different destination.";

            if (error.includes("ZERO_RESULTS")) {
                errorMsg = "No route available";
                speakMsg = "No walking route found. The destination might be too far or unreachable on foot.";
            } else if (error.includes("NOT_FOUND")) {
                errorMsg = "Location not found";
                speakMsg = "I couldn't find that location. Please try saying the full address or a well-known landmark.";
            } else if (error.includes("Geocode")) {
                errorMsg = "Invalid destination";
                speakMsg = "I couldn't understand that destination. Please try again with a full address or landmark name.";
            }

            setNavState({ step: 'error', destination, error: errorMsg });
            speak(speakMsg);
        }
    }, [currentLocation, locationReady, speak]);

    // Handle Voice Commands
    const handleVoiceCommand = useCallback(async (command: string) => {
        console.log("Received voice command:", command);

        // Optimistic update to show we are processing
        setNavState(prev => ({ ...prev, step: 'processing' }));

        // First check for simple local commands like "Start" if we are already ready
        // This avoids network latency for simple start confirmations
        const lowerCommand = command.toLowerCase().trim();
        if (navState.step === 'ready' && (lowerCommand.includes('start') || lowerCommand.includes('begin') || lowerCommand.includes('go'))) {
            setNavState(prev => ({ ...prev, step: 'navigating' }));
            setCurrentStepIndex(0);
            setLastAnnouncedStep(-1);
            speak("Starting navigation.");
            return;
        }

        // Helper to extract destination from a generic object
        const extractDestination = (payload: Record<string, unknown> | undefined): string => {
            if (!payload) return '';
            const keys = ['destination', 'location', 'address', 'place', 'geo-city', 'geo_city', 'city', 'sys.any'];
            for (const key of keys) {
                const val = (payload as any)[key];
                if (!val) continue;
                if (typeof val === 'string') return val;
                if (typeof val === 'object' && val !== null) {
                    if ((val as any).original) return (val as any).original;
                    if ((val as any).value) return (val as any).value;
                    return JSON.stringify(val);
                }
            }
            return '';
        };

        try {
            const response = await askVapi(command);
            console.log("Vapi Result:", response);

            const destination =
                response.destination ||
                extractDestination(response.args) ||
                extractDestination(response.parameters) ||
                extractDestination(response.data);

            const responseText =
                (response as any).text ||
                (response as any).message ||
                (response as any).reply ||
                '';

            // Play agent audio if provided, otherwise speak text if any
            if (response.outputAudio) {
                try {
                    const audioUrl = `data:audio/mp3;base64,${response.outputAudio}`;
                    const audio = new Audio(audioUrl);

                    if (audioRef.current) {
                        audioRef.current.pause();
                        audioRef.current = null;
                    }

                    audioRef.current = audio;
                    await audio.play();
                } catch (audioErr) {
                    console.error("Failed to play Vapi audio:", audioErr);
                    if (responseText) speak(responseText);
                }
            } else if (responseText) {
                speak(responseText);
            }

            // If Vapi returned a destination, proceed to route calculation
            if (destination && destination.length > 2) {
                console.log("Extracted destination from Vapi:", destination);
                setNavState(prev => ({ ...prev, step: 'processing', destination }));
                calculateRoute(destination);
                return;
            }

            // If no destination, but user said start and we are ready, start navigation
            if (navState.step === 'ready' && (responseText.toLowerCase().includes('start') || responseText.toLowerCase().includes('begin'))) {
                setNavState(prev => ({ ...prev, step: 'navigating' }));
                setCurrentStepIndex(0);
                setLastAnnouncedStep(-1);
                speak("Starting navigation.");
                return;
            }

            // No action taken; reset state to idle/ready
            setNavState(prev => {
                if (prev.step === 'processing') {
                    if (prev.destination) return { ...prev, step: 'ready' };
                    return { ...prev, step: 'idle' };
                }
                return prev;
            });

        } catch (error) {
            console.error("Vapi integration error:", error);
            const errMsg = error instanceof Error ? error.message : String(error);

            // Fallback to local regex matching
            const prefixes = ["navigate to", "go to", "take me to", "find", "search for", "directions to"];
            let fallbackDest = '';
            for (const prefix of prefixes) {
                if (command.toLowerCase().startsWith(prefix)) {
                    fallbackDest = command.toLowerCase().slice(prefix.length).trim();
                    break;
                }
            }

            if (fallbackDest) {
                speak(`I'll try to find ${fallbackDest}`);
                setNavState(prev => ({ ...prev, step: 'processing', destination: fallbackDest }));
                calculateRoute(fallbackDest);
            } else {
                speak("I'm having trouble understanding. Please try again.");
                setNavState(prev => ({ ...prev, step: 'idle', error: `Agent Error: ${errMsg}` }));
            }
        }

    }, [navState.step, speak, calculateRoute]);

    // Speech Recognition
    const startListening = useCallback(() => {
        if (recognitionRef.current) {
            try {
                recognitionRef.current.onend = null;
                recognitionRef.current.abort();
            } catch (e) { }
        }

        listenRetryRef.current = 0;

        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            speak("Voice recognition is not supported on this browser.");
            return;
        }

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            setNavState(prev => ({ ...prev, step: 'listening', error: undefined }));
            setInstruction("Listening... Where do you want to go?");
        };

        recognition.onresult = (event: any) => {
            const transcript = event.results[0][0].transcript;
            if (transcript) {
                handleVoiceCommand(transcript);
            }
        };

        recognition.onerror = (event: any) => {
            console.error("Speech Error:", event.error);
            let speakMsg = "I didn't catch that. Tap to try again.";

            if (event.error === 'not-allowed') {
                speakMsg = "Please allow microphone access in your browser settings.";
            } else if (event.error === 'no-speech') {
                speakMsg = "I didn't hear anything. Listening again.";
                // Automatically retry once on silence so users don't need to tap again
                if (listenRetryRef.current < 1) {
                    listenRetryRef.current += 1;
                    setTimeout(() => {
                        startListening();
                    }, 400);
                    speak(speakMsg);
                    return;
                }
            } else if (event.error === 'network') {
                speakMsg = "Please check your internet connection.";
            }

            setNavState(prev => ({ ...prev, step: 'idle', error: event.error }));
            speak(speakMsg);
        };

        recognition.onend = () => {
            if (navState.step === 'listening') {
                setNavState(prev => ({ ...prev, step: 'idle' }));
            }
        };

        try {
            recognition.start();
        } catch (e) {
            console.error("Recognition start failed:", e);
        }
    }, [speak, handleVoiceCommand, navState.step]);

    // Handle Screen Interaction
    const handleInteraction = () => {
        if (navState.step === 'listening') return;

        if (navState.step === 'ready') {
            setNavState(prev => ({ ...prev, step: 'navigating' }));
            setCurrentStepIndex(0);
            setLastAnnouncedStep(-1);
            speak("Starting navigation. I will guide you with turn by turn directions.");
        } else if (navState.step === 'navigating') {
            if (directionsResponse) {
                const steps = directionsResponse.routes[0].legs[0].steps;
                if (steps[currentStepIndex]) {
                    const clean = steps[currentStepIndex].instructions.replace(/<[^>]*>/g, '');
                    speak(`Current instruction: ${clean}`);
                }
            }
        } else {
            speak("Where do you want to go?", () => {
                startListening();
            });
        }
    };

    return (
        <div className="relative w-full h-screen bg-black" onClick={handleInteraction}>
            <div id="map" className="absolute inset-0 z-0 opacity-50 bg-neutral-900 flex items-center justify-center" style={mapContainerStyle}>
                {!VITE_GOOGLE_MAPS_API_KEY && (
                    <div className="text-white/20 font-bold text-4xl uppercase tracking-widest">Demo Mode</div>
                )}
            </div>

            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 text-center pointer-events-none">
                <div className="mb-8 p-8 rounded-full bg-indigo-600/80 shadow-[0_0_50px_rgba(79,70,229,0.5)] animate-pulse">
                    {navState.step === 'listening' ? (
                        <Mic className="w-24 h-24 text-white" />
                    ) : navState.step === 'navigating' ? (
                        <Navigation className="w-24 h-24 text-white" />
                    ) : navState.step === 'processing' ? (
                        <Loader2 className="w-24 h-24 text-white animate-spin" />
                    ) : (
                        <Navigation className="w-24 h-24 text-white opacity-50" />
                    )}
                </div>

                <div className="bg-black/80 p-6 rounded-2xl backdrop-blur-md border border-white/10 max-w-2xl">
                    <h1 className="text-2xl md:text-4xl font-black text-white mb-2">
                        {navState.step === 'listening' ? "Listening..." :
                            navState.step === 'processing' ? "Finding route..." :
                                navState.step === 'ready' ? "Ready! Tap or say 'Start'" :
                                    navState.step === 'navigating' ? instruction :
                                        !locationReady ? "Getting your location..." :
                                            "Tap to Start Navigation"}
                    </h1>

                    {navState.step === 'idle' && locationReady && (
                        <p className="text-gray-400 text-sm mt-2">Say your destination when prompted</p>
                    )}
                </div>

                {navState.error && (
                    <div className="mt-4 bg-red-900/80 text-white px-6 py-3 rounded-full flex items-center gap-2">
                        <AlertCircle size={20} />
                        <span>{navState.error}</span>
                    </div>
                )}

                {navState.step === 'navigating' && directionsResponse && (
                    <div className="mt-4 bg-indigo-900/80 text-white px-6 py-3 rounded-full">
                        <span>Step {currentStepIndex + 1} of {directionsResponse.routes[0].legs[0].steps.length}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default BlindNavigation;