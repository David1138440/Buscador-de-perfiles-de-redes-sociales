import React, { useState, useCallback, useMemo } from 'react';
import { analyzeImageWithGemini } from '../services/geminiService';

type Status = 'idle' | 'loading' | 'success' | 'error';

const Spinner: React.FC = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

const UploadIcon: React.FC = () => (
    <svg className="w-12 h-12 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
    </svg>
);

export const ImageAnalyzer: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<Status>('idle');
    const [result, setResult] = useState<string>('');
    const [error, setError] = useState<string>('');
    
    const previewUrl = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = event.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setStatus('idle');
            setResult('');
            setError('');
        }
    };

    const handleAnalyze = useCallback(async () => {
        if (!file) return;

        setStatus('loading');
        setResult('');
        setError('');

        try {
            const analysisResult = await analyzeImageWithGemini(file);
            setResult(analysisResult);
            setStatus('success');
        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            setStatus('error');
        }
    }, [file]);

    const handleClear = () => {
        setFile(null);
        setStatus('idle');
        setResult('');
        setError('');
        // Also clear the file input value
        const fileInput = document.getElementById('file-upload') as HTMLInputElement;
        if(fileInput) fileInput.value = '';
    };
    
    return (
        <div className="w-full max-w-4xl p-6 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg">
            {!file ? (
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="file-upload" className="flex flex-col items-center justify-center w-full h-64 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer bg-slate-900/50 hover:bg-slate-800/50 transition-colors">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <UploadIcon />
                            <p className="mb-2 text-sm text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                            <p className="text-xs text-gray-500">PNG, JPG, GIF, WEBP</p>
                        </div>
                        <input id="file-upload" type="file" className="hidden" onChange={handleFileChange} accept="image/png, image/jpeg, image/gif, image/webp" />
                    </label>
                </div> 
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex flex-col items-center">
                        <div className="w-full aspect-square rounded-lg overflow-hidden border border-slate-700 mb-4">
                            {previewUrl && <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />}
                        </div>
                        <div className="w-full flex space-x-2">
                             <button onClick={handleAnalyze} disabled={status === 'loading'} className="flex-grow inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500 disabled:bg-indigo-500/50 disabled:cursor-not-allowed transition-all">
                                {status === 'loading' && <Spinner />}
                                {status === 'loading' ? 'Analyzing...' : 'Analyze Image'}
                            </button>
                             <button onClick={handleClear} className="px-4 py-2 border border-slate-600 text-sm font-medium rounded-md text-slate-300 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-slate-500 transition-colors">
                                Clear
                            </button>
                        </div>
                    </div>

                    <div className="flex flex-col">
                        <h2 className="text-xl font-semibold mb-2 text-slate-200">Analysis Result</h2>
                         <div className="prose prose-invert prose-p:text-slate-300 prose-headings:text-slate-100 flex-grow p-4 bg-slate-900/70 rounded-md border border-slate-700 min-h-[200px] overflow-y-auto">
                           {status === 'loading' && <p className="text-slate-400 animate-pulse">Gemini is thinking...</p>}
                           {status === 'success' && <p>{result}</p>}
                           {status === 'error' && <p className="text-red-400">Error: {error}</p>}
                           {status === 'idle' && <p className="text-slate-500">Click "Analyze Image" to see results here.</p>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
