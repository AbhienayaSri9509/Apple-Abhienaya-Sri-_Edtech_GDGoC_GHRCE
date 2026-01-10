import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import SignToTextPage from './pages/SignToTextPage';
import TextToSignPage from './pages/TextToSignPage';
import DirectionPage from './pages/DirectionPage';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/sign-to-text" element={<SignToTextPage />} />
        <Route path="/text-to-sign" element={<TextToSignPage />} />
        <Route path="/directions" element={<DirectionPage />} />
      </Routes>
    </Router>
  );
};

export default App;