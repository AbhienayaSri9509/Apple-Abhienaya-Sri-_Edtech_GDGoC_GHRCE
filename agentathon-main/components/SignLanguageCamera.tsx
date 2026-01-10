import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Video, Square, Camera } from 'lucide-react';

interface SignLanguageCameraProps {
  onSignDetected: (text: string) => void;
  onProcessingChange?: (isProcessing: boolean) => void;
  language?: 'en' | 'ta';
}

const SignLanguageCamera: React.FC<SignLanguageCameraProps> = ({ onSignDetected, onProcessingChange, language = 'en' }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const diffCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const prevFrameDataRef = useRef<ImageData | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [flash, setFlash] = useState(false);
  const [detectedText, setDetectedText] = useState<string>('');
  const [permissionStatus, setPermissionStatus] = useState<'idle' | 'prompt' | 'granted' | 'denied'>('idle');
  const [isRequestingCamera, setIsRequestingCamera] = useState(false);

  // Stop everything (Stream + Capture)
  const stopEverything = useCallback(() => {
    console.log('Stopping everything...');
    setIsCapturing(false);
    setIsStreaming(false);

    // clear interval
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    // stop tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const requestCameraAccess = useCallback(async (): Promise<boolean> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is not supported in this browser.");
      setPermissionStatus('denied');
      return false;
    }

    setIsRequestingCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      streamRef.current = stream;
      setIsStreaming(true); // triggering re-render to show <video>
      setError(null);
      setPermissionStatus('granted');
      return true;
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("Camera access denied or not available. Please allow permission and try again.");
      setIsStreaming(false);
      setPermissionStatus('denied');
      return false;
    } finally {
      setIsRequestingCamera(false);
    }
  }, []);

  // Effect to attach stream to video element once it's mounted
  useEffect(() => {
    if (isStreaming && videoRef.current && streamRef.current) {
      console.log("Attaching stream to video element");
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(e => console.error("Error playing video:", e));
    }
  }, [isStreaming]);

  const captureAndProcessFrame = useCallback(async () => {
    // 1. Guard: If already processing a frame, skip this tick completely.
    if (!videoRef.current || isProcessing || !canvasRef.current) return;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video.videoWidth || !video.videoHeight) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // 2. Motion Filter: Check if frame matches previous one to avoid sending duplicates.
      let isMeaningfulChange = true;
      try {
        const diffCanvas = diffCanvasRef.current ?? document.createElement('canvas');
        diffCanvas.width = 64; // Small size for performance
        diffCanvas.height = 48;
        if (!diffCanvasRef.current) diffCanvasRef.current = diffCanvas;

        const dctx = diffCanvas.getContext('2d');
        if (dctx) {
          dctx.drawImage(video, 0, 0, diffCanvas.width, diffCanvas.height);
          const current = dctx.getImageData(0, 0, diffCanvas.width, diffCanvas.height);
          const prev = prevFrameDataRef.current;

          if (prev) {
            let changed = 0;
            const total = current.data.length / 4;
            for (let i = 0; i < current.data.length; i += 4) {
              const dr = Math.abs(current.data[i] - prev.data[i]);
              const dg = Math.abs(current.data[i + 1] - prev.data[i + 1]);
              const db = Math.abs(current.data[i + 2] - prev.data[i + 2]);
              if (dr + dg + db > 45) changed++; // Pixel logic kept similar to before
            }
            const changeRatio = changed / total;
            // Adjustable threshold: >4% pixels changed
            isMeaningfulChange = changeRatio > 0.04;
          }
          prevFrameDataRef.current = current;
        }
      } catch (motionErr) {
        console.warn('Motion check failed, proceeding with frame:', motionErr);
      }

      if (!isMeaningfulChange) {
        // Skip sending this frame, but we are effectively "done" with this tick.
        return;
      }

      // 3. Processing: Send to Gemini (optimize image size for faster API response)
      // Reduce image size to speed up API calls - Gemini works well with smaller images
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 320; // Half resolution for faster processing
      tempCanvas.height = 240;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      }
      const frameData = tempCanvas.toDataURL('image/jpeg', 0.7); // Lower quality = smaller payload
      const base64Data = frameData.split(',')[1];

      // Flash effect to indicate *capture*
      setFlash(true);
      setTimeout(() => setFlash(false), 150);

      setIsProcessing(true);
      onProcessingChange?.(true);

      const { processSignLanguageImage } = await import('../services/signLanguage');

      try {
        const text = await processSignLanguageImage(base64Data, language);

        // 4. Response Handling
        if (text && text.trim()) {
          setDetectedText(text);
          setError(null);
          onSignDetected(text); // Triggers TTS in parent
        }
      } catch (procErr: any) {
        console.error('Error processing frame with Gemini:', procErr);
        // We do NOT show error usage to user to keep UI clean, just log it.
      } finally {
        // Only NOW strictly allow the next frame to be picked up by the interval ticks
        setIsProcessing(false);
        onProcessingChange?.(false);
      }

    } catch (err) {
      console.error('Error capturing frame:', err);
      setIsProcessing(false);
      onProcessingChange?.(false);
    }
  }, [isProcessing, language, onSignDetected, onProcessingChange]);

  // Start Capture Loop
  const startCapturing = useCallback(() => {
    if (!isStreaming) return;

    console.log('Starting continuous sign capture...');
    setIsCapturing(true);
    setDetectedText('');

    // Clear any existing
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);

    // Initial capture immediately
    captureAndProcessFrame();

    // Capture every 2 seconds. 
    // Note: detailed logic ensures we skip if isProcessing is true.
    captureIntervalRef.current = window.setInterval(() => {
      captureAndProcessFrame();
    }, 2000);

  }, [isStreaming, captureAndProcessFrame]);


  // Master Toggle Function
  const toggleCameraSystem = useCallback(async () => {
    if (isStreaming) {
      // Turn OFF
      stopEverything();
    } else {
      // Turn ON
      const granted = await requestCameraAccess();
      if (granted) {
        // Automatically start capturing after a brief delay to let video warm up
        setTimeout(() => {
          // We can trigger the capture state, but we need to ensure the streaming state is reflected.
          // Using a small timeout usually works fine for UX here.
          setIsCapturing(true);
        }, 500);
      }
    }
  }, [isStreaming, requestCameraAccess, stopEverything]);

  // Effect to manage the interval based on isCapturing state changes
  // This ensures if we set setIsCapturing(true) elsewhere, it starts the loop.
  useEffect(() => {
    if (isCapturing && isStreaming) {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);

      // Run one immediately? Maybe not, startCapturing usually called it. 
      // But if we just toggled isCapturing from UI:
      // captureAndProcessFrame(); 

      captureIntervalRef.current = window.setInterval(captureAndProcessFrame, 2000);
    } else {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    }
  }, [isCapturing, isStreaming, captureAndProcessFrame]);


  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopEverything();
    };
  }, [stopEverything]);

  // Keyboard 'B' listener
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'b' && !e.repeat) { // Ignore repeat events
        console.log("'B' Key Pressed");
        toggleCameraSystem();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [toggleCameraSystem]);


  return (
    <div className="fixed left-4 top-20 z-50 transition-all duration-300">

      {/* Main Container */}
      <div className={`
        relative backdrop-blur-md rounded-xl border shadow-2xl overflow-hidden transition-all duration-300
        ${isStreaming ? 'w-48 bg-black/80 border-white/20' : 'w-auto bg-transparent border-transparent shadow-none'}
      `}>

        {/* Header / Status - Only visible when Active */}
        {isStreaming && (
          <div className="absolute top-2 left-2 z-20 flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isCapturing ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
            <span className="text-[10px] font-bold text-white shadow-black drop-shadow-md">
              {isProcessing ? 'ANALYZING...' : isCapturing ? 'LIVE (v2.1)' : 'READY'}
            </span>
          </div>
        )}

        {/* Video Feed */}
        {isStreaming && (
          <div className="relative aspect-[4/3] w-full bg-black">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover mirror-mode scale-x-[-1]"
            />

            {/* Flash Overlay */}
            <div className={`absolute inset-0 bg-white pointer-events-none transition-opacity duration-150 ${flash ? 'opacity-30' : 'opacity-0'}`} />

            {/* Overlay for instructions if needed */}
          </div>
        )}

        {/* Capture Controls */}
        {isStreaming && (
          <div className="flex items-center justify-between gap-2 bg-black/70 px-3 py-2 border-t border-white/10">
            <button
              type="button"
              onClick={startCapturing}
              disabled={isCapturing || isProcessing}
              className={`flex-1 text-[11px] font-semibold py-1 rounded ${!isCapturing && !isProcessing
                ? 'bg-green-600 hover:bg-green-500 text-white'
                : 'bg-green-900/40 text-white/50 cursor-not-allowed'
                }`}
            >
              Start Capture
            </button>
            <button
              type="button"
              onClick={() => setIsCapturing(false)}
              disabled={!isCapturing}
              className={`flex-1 text-[11px] font-semibold py-1 rounded ${isCapturing
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-red-900/40 text-white/50 cursor-not-allowed'
                }`}
            >
              Stop
            </button>
          </div>
        )}

        {/* Error State intentionally hidden to keep camera unobtrusive */}

      </div>

      {/* Guide / Hint - Visible when NOT streaming or unobtrusive */}
      {!isStreaming && (
        <div className="flex items-center gap-3 bg-black/40 backdrop-blur-sm px-4 py-2 rounded-full border border-white/10 hover:bg-black/60 transition-colors group cursor-pointer" onClick={toggleCameraSystem}>
          <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center group-hover:scale-110 transition-transform">
            <Camera size={16} className="text-white" />
          </div>
          <div className="flex flex-col">
            <span className="text-xs font-bold text-white">Sign Camera</span>
            <span className="text-[10px] text-gray-300">Press <kbd className="bg-gray-700 px-1 rounded text-white font-mono">B</kbd> to start</span>
          </div>
        </div>
      )}

      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default SignLanguageCamera;

