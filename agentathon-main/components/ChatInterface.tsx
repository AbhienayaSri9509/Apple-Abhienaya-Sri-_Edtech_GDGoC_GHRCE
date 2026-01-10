import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, User, Cpu } from 'lucide-react';
import { Message, ChatStatus } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  status: ChatStatus;
  onSendMessage: (text: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages, status, onSendMessage }) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && status !== ChatStatus.LOADING) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between z-10">
      
      {/* Header */}
      <div className="w-full p-6 flex justify-between items-start bg-gradient-to-b from-black/50 to-transparent pointer-events-auto">
        <div>
           <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-violet-500">
             Lumina
           </h1>
           <p className="text-xs text-white/60">Virtual Companion</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/10">
           <div className={`w-2 h-2 rounded-full ${status === ChatStatus.LOADING ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`} />
           <span className="text-xs font-medium text-white/80">
             {status === ChatStatus.LOADING ? 'Thinking...' : 'Online'}
           </span>
        </div>
      </div>

      {/* Main Chat Area - Centered for visibility but offset to not block avatar too much */}
      <div className="flex-1 flex flex-col items-center justify-end pb-24 w-full max-w-2xl mx-auto px-4">
        
        {/* Message History Container */}
        <div className="w-full max-h-[40vh] overflow-y-auto scrollbar-hide flex flex-col gap-3 pointer-events-auto mask-image-gradient p-2">
           {messages.length === 0 && (
             <div className="text-center text-white/40 italic py-4">
               Say hello to Lumina...
             </div>
           )}
           {messages.map((msg, idx) => (
             <div 
               key={idx} 
               className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
             >
               <div className={`
                 max-w-[80%] rounded-2xl px-4 py-3 backdrop-blur-md border border-white/10 text-sm shadow-lg
                 ${msg.role === 'user' 
                   ? 'bg-pink-600/80 text-white rounded-br-none' 
                   : 'bg-indigo-900/60 text-indigo-50 rounded-bl-none'}
               `}>
                 <div className="flex items-center gap-2 mb-1 opacity-50 text-[10px] uppercase tracking-wider font-bold">
                    {msg.role === 'user' ? <User size={10} /> : <Cpu size={10} />}
                    {msg.role === 'user' ? 'You' : 'Lumina'}
                 </div>
                 {msg.content}
               </div>
             </div>
           ))}
           <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="w-full mt-4 pointer-events-auto">
           <form onSubmit={handleSubmit} className="relative group">
             <div className="absolute -inset-1 bg-gradient-to-r from-pink-600 to-violet-600 rounded-xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
             <div className="relative flex items-center bg-black/60 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden shadow-2xl">
               <input
                 type="text"
                 value={inputText}
                 onChange={(e) => setInputText(e.target.value)}
                 placeholder="Type a message..."
                 className="flex-1 bg-transparent border-none outline-none px-6 py-4 text-white placeholder-white/40 font-medium"
                 disabled={status === ChatStatus.LOADING}
               />
               <button 
                 type="submit"
                 disabled={!inputText.trim() || status === ChatStatus.LOADING}
                 className="px-6 py-2 m-2 bg-white/10 hover:bg-white/20 rounded-lg text-pink-400 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
               >
                 {status === ChatStatus.LOADING ? <Sparkles className="animate-spin" size={20} /> : <Send size={20} />}
               </button>
             </div>
           </form>
        </div>

      </div>
    </div>
  );
};

export default ChatInterface;