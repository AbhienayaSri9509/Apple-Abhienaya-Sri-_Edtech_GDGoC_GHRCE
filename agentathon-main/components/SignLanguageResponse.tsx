import React from 'react';
import { Hand } from 'lucide-react';

interface SignLanguageResponseProps {
  detectedText: string;
}

const SignLanguageResponse: React.FC<SignLanguageResponseProps> = ({ detectedText }) => {
  if (!detectedText) {
    return (
      <div className="fixed right-4 top-20 z-50 bg-black/60 backdrop-blur-md rounded-lg p-2 border border-white/20 shadow-xl w-48">
        <div className="flex items-center gap-1.5 mb-1.5">
          <Hand size={12} className="text-indigo-400" />
          <h3 className="text-white text-xs font-bold">Sign Response</h3>
        </div>
        <div className="text-center py-4">
          <p className="text-white/40 text-xs italic">No sign detected yet</p>
          <p className="text-white/30 text-[10px] mt-1.5">Press 'B' to start</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed right-4 top-20 z-50 bg-black/60 backdrop-blur-md rounded-lg p-2 border border-white/20 shadow-xl w-48">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Hand size={12} className="text-indigo-400" />
        <h3 className="text-white text-xs font-bold">Sign Response</h3>
      </div>
      
      <div className="bg-indigo-900/60 rounded-lg p-2 border border-indigo-500/30">
        <p className="text-[10px] text-indigo-200 font-semibold mb-1 uppercase tracking-wider">Detected:</p>
        <p className="text-sm text-white font-bold break-words">{detectedText}</p>
      </div>
      
      <div className="mt-1.5 pt-1.5 border-t border-white/10">
        <p className="text-[10px] text-white/50">
          Text will be sent to Lumina.
        </p>
      </div>
    </div>
  );
};

export default SignLanguageResponse;

