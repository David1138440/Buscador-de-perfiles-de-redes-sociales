import React, { useState, useRef, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';


type RecordingStatus = 'idle' | 'permission-pending' | 'ready' | 'recording' | 'finished';

// Icons for UI elements
const CameraIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.55a1 1 0 011.45.89V18a1 1 0 01-1.45.89L15 16M4 6h10a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V8a2 2 0 012-2z" />
    </svg>
);

const Spinner = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

export const VideoRecorder: React.FC = () => {
    const [status, setStatus] = useState<RecordingStatus>('idle');
    const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
    const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [duration, setDuration] = useState<number>(1); // Default duration 1 minute
    const [elapsedTime, setElapsedTime] = useState(0);

    // State for face detection
    const [detector, setDetector] = useState<faceDetection.FaceDetector | null>(null);
    const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
    const [isDetectionEnabled, setIsDetectionEnabled] = useState<boolean>(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const recordedChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<number | null>(null);
    const stopwatchRef = useRef<number | null>(null);
    const detectionLoopRef = useRef<number | null>(null);

    // Effect to load the face detection model
    useEffect(() => {
        const loadModel = async () => {
            if (status === 'ready' && !detector && !isModelLoading) {
                setIsModelLoading(true);
                try {
                    const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
                    // FIX: Use `MediaPipeFaceDetectorTfjsModelConfig` instead of `MediaPipeFaceDetectorModelConfig`
                    const detectorConfig: faceDetection.MediaPipeFaceDetectorTfjsModelConfig = {
                        runtime: 'mediapipe',
                        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4.1646425229',
                    };
                    const createdDetector = await faceDetection.createDetector(model, detectorConfig);
                    setDetector(createdDetector);
                } catch (e) {
                    console.error("Failed to load face detection model", e);
                    setError("No se pudo cargar el modelo de detección de rostros.");
                } finally {
                    setIsModelLoading(false);
                }
            }
        };
        loadModel();
    }, [status, detector, isModelLoading]);

    // Effect to run the detection loop
    useEffect(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');

        const detectFaces = async () => {
            if (isDetectionEnabled && detector && video && video.readyState >= 3 && ctx && canvas) {
                const faces = await detector.estimateFaces(video, { flipHorizontal: false });

                // Set canvas size to match video display size
                canvas.width = video.clientWidth;
                canvas.height = video.clientHeight;

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Scale factors to draw on canvas
                const scaleX = canvas.width / video.videoWidth;
                const scaleY = canvas.height / video.videoHeight;

                for (const face of faces) {
                    // Draw bounding box
                    ctx.strokeStyle = '#00FF00'; // Green
                    ctx.lineWidth = 2;
                    const { xMin, yMin, width, height } = face.box;
                    ctx.strokeRect(xMin * scaleX, yMin * scaleY, width * scaleX, height * scaleY);

                    // Draw keypoints (facial features)
                    ctx.fillStyle = '#00FFFF'; // Cyan
                    face.keypoints.forEach(keypoint => {
                        ctx.beginPath();
                        ctx.arc(keypoint.x * scaleX, keypoint.y * scaleY, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            }
            detectionLoopRef.current = requestAnimationFrame(detectFaces);
        };

        if (isDetectionEnabled) {
            detectFaces();
        } else {
             if (detectionLoopRef.current) {
                cancelAnimationFrame(detectionLoopRef.current);
            }
            ctx?.clearRect(0, 0, canvas?.width ?? 0, canvas?.height ?? 0);
        }

        return () => {
            if (detectionLoopRef.current) {
                cancelAnimationFrame(detectionLoopRef.current);
            }
        };
    }, [isDetectionEnabled, detector, status]);


    const handleRequestPermission = useCallback(async () => {
        setStatus('permission-pending');
        setError(null);

        const videoConstraints = { facingMode: 'environment' };

        try {
            let stream: MediaStream;
            try {
                 stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: true });
            } catch (e) {
                console.warn("Could not get environment camera, trying default.", e);
                stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            }

            setMediaStream(stream);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setStatus('ready');
        } catch (err) {
            console.error("Error accessing media devices.", err);
            setError("No se pudo acceder a la cámara y al micrófono. Por favor, comprueba los permisos en la configuración de tu navegador.");
            setStatus('idle');
        }
    }, []);

    const startStopwatch = () => {
        setElapsedTime(0);
        stopwatchRef.current = window.setInterval(() => {
            setElapsedTime(prev => prev + 1);
        }, 1000);
    };

    const stopStopwatch = () => {
        if (stopwatchRef.current) {
            clearInterval(stopwatchRef.current);
            stopwatchRef.current = null;
        }
    };
    
    const handleStopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            if (timerRef.current) {
                clearTimeout(timerRef.current);
                timerRef.current = null;
            }
        }
    }, []);

    const handleStartRecording = useCallback(() => {
        if (!mediaStream) return;
        
        setStatus('recording');
        setIsDetectionEnabled(false); // Disable detection during recording to save performance
        recordedChunksRef.current = [];
        const recorder = new MediaRecorder(mediaStream);
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunksRef.current.push(event.data);
            }
        };

        recorder.onstop = () => {
            const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            setRecordedVideoUrl(url);
            setStatus('finished');
            stopStopwatch();
            mediaStream.getTracks().forEach(track => track.stop());
            setMediaStream(null);
        };

        recorder.start();
        startStopwatch();

        if (duration > 0) {
            timerRef.current = window.setTimeout(() => {
                handleStopRecording();
            }, duration * 60 * 1000);
        }

    }, [mediaStream, duration, handleStopRecording]);
    
    const handleDownload = () => {
        if (!recordedVideoUrl) return;
        const a = document.createElement('a');
        a.href = recordedVideoUrl;
        a.download = `recording-${new Date().toISOString()}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    const handleNewRecording = () => {
        if (recordedVideoUrl) {
            URL.revokeObjectURL(recordedVideoUrl);
        }
        setRecordedVideoUrl(null);
        setStatus('idle');
        setError(null);
        setElapsedTime(0);
    };
    
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
        const secs = (seconds % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
    };

    return (
        <div className="w-full max-w-4xl p-6 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg flex flex-col items-center">
            {status === 'idle' && (
                 <button onClick={handleRequestPermission} className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500 transition-all">
                    <CameraIcon />
                    Iniciar Cámara
                </button>
            )}

            {status === 'permission-pending' && <div className="flex items-center text-lg"><Spinner /> Solicitando permisos...</div>}
            
            {error && <p className="text-red-400 mt-4 text-center">{error}</p>}
            
            {(status === 'ready' || status === 'recording') && (
                <div className="w-full flex flex-col items-center">
                    <div className="w-full aspect-video rounded-lg overflow-hidden border-2 border-slate-700 bg-black mb-4 relative">
                        <video ref={videoRef} autoPlay muted className="w-full h-full object-cover"></video>
                        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none"></canvas>
                        {status === 'recording' && (
                             <div className="absolute top-2 left-2 flex items-center bg-black/50 p-2 rounded-lg" aria-live="polite">
                                <span className="h-3 w-3 rounded-full bg-red-500 animate-pulse mr-2"></span>
                                <span className="text-white font-mono">{formatTime(elapsedTime)}</span>
                            </div>
                        )}
                    </div>

                    <div className="w-full flex flex-col sm:flex-row items-center justify-center gap-4">
                         {status === 'ready' && (
                            <div className="flex items-center gap-3 bg-slate-900/50 p-2 rounded-md">
                                <label htmlFor="detection-toggle" className="text-slate-300 font-medium">Detección Facial:</label>
                                <div className="flex items-center">
                                    <input
                                        id="detection-toggle"
                                        type="checkbox"
                                        className="form-checkbox h-5 w-5 text-indigo-600 bg-slate-800 border-slate-600 rounded focus:ring-indigo-500 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
                                        checked={isDetectionEnabled}
                                        onChange={(e) => setIsDetectionEnabled(e.target.checked)}
                                        disabled={isModelLoading || !detector}
                                    />
                                    {isModelLoading && <Spinner />}
                                </div>
                            </div>
                        )}

                        {status === 'ready' && (
                             <div className="flex items-center gap-2">
                                <label htmlFor="duration" className="text-slate-300">Duración (min):</label>
                                <input 
                                    id="duration"
                                    type="number" 
                                    value={duration}
                                    onChange={(e) => setDuration(Math.max(1, parseInt(e.target.value, 10)))}
                                    min="1"
                                    className="w-20 bg-slate-900 border border-slate-600 rounded-md p-2 text-center"
                                />
                            </div>
                        )}
                       
                        {status === 'ready' && (
                            <button onClick={handleStartRecording} className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-md font-semibold transition-colors">Iniciar Grabación</button>
                        )}
                        {status === 'recording' && (
                            <button onClick={handleStopRecording} className="px-6 py-3 bg-red-600 hover:bg-red-700 rounded-md font-semibold transition-colors">Detener Grabación</button>
                        )}
                    </div>
                </div>
            )}
            
            {status === 'finished' && recordedVideoUrl && (
                 <div className="w-full flex flex-col items-center">
                     <h2 className="text-xl font-semibold mb-2 text-slate-200">Grabación Completa</h2>
                     <video src={recordedVideoUrl} controls className="w-full aspect-video rounded-lg border-2 border-slate-700 mb-4"></video>
                     <div className="flex gap-4">
                        <button onClick={handleDownload} className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-md font-semibold transition-colors">Descargar</button>
                        <button onClick={handleNewRecording} className="px-5 py-2 bg-slate-600 hover:bg-slate-700 rounded-md font-semibold transition-colors">Grabar Otro</button>
                     </div>
                 </div>
            )}
        </div>
    );
};