import React from 'react';
import { VideoRecorder } from './components/VideoRecorder';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-900 text-gray-200 flex flex-col items-center p-4 sm:p-6 lg:p-8">
      <header className="w-full max-w-4xl text-center mb-8">
        <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-600">
          Real-time Video Recorder
        </h1>
        <p className="mt-2 text-lg text-slate-400">
          Record video and audio directly from your browser.
        </p>
      </header>
      <main className="w-full flex justify-center">
        <VideoRecorder />
      </main>
    </div>
  );
};

export default App;
