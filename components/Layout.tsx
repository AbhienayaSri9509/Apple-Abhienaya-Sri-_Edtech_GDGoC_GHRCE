import React, { ReactNode } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

interface LayoutProps {
    children: ReactNode;
    showBackButton?: boolean;
}

const Layout: React.FC<LayoutProps> = ({ children, showBackButton = true }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const isHome = location.pathname === '/';

    return (
        <div className="relative w-full h-screen bg-slate-900 overflow-hidden font-sans text-white">
            {/* Background Gradient - Consistent across app */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900 via-purple-900 to-black opacity-40 z-0 pointer-events-none" />

            {/* Navigation Header */}
            {showBackButton && !isHome && (
                <button
                    onClick={() => navigate('/')}
                    className="absolute top-4 left-4 z-50 p-2 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm transition-all text-white border border-white/10"
                    aria-label="Go Back"
                >
                    <ArrowLeft size={24} />
                </button>
            )}

            {/* Content */}
            <div className="relative z-10 w-full h-full">
                {children}
            </div>
        </div>
    );
};

export default Layout;
