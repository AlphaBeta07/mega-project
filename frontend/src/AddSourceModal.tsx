import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, UploadCloud, FileText, PlaySquare, Link2,
  Trash2, FileAudio, FileVideo, FileSpreadsheet, Loader2, CheckCircle, Image as ImageIcon
} from 'lucide-react';

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (newSources: any[]) => void;
  backendUrl: string;
}

const ACCEPTED_FORMATS = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'text/markdown': ['.md'],
  'application/json': ['.json'],
  'text/html': ['.html'],
  'application/xml': ['.xml'],
  'audio/mpeg': ['.mp3'],
  'audio/wav': ['.wav'],
  'audio/x-m4a': ['.m4a'],
  'audio/ogg': ['.ogg', '.opus'],
  'audio/aac': ['.aac'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/webp': ['.webp'],
  'image/gif': ['.gif']
};

export default function AddSourceModal({ isOpen, onClose, onUploadSuccess, backendUrl }: AddSourceModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [urlInput, setUrlInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatusText, setUploadStatusText] = useState("Processing...");
  const [activeTab, setActiveTab] = useState<'files' | 'url'>('files');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_FORMATS,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (['mp3', 'wav', 'm4a', 'ogg', 'opus', 'aac'].includes(ext || '')) return <FileAudio size={20} className="text-blue-400" />;
    if (['mp4', 'mov'].includes(ext || '')) return <FileVideo size={20} className="text-purple-400" />;
    if (['csv', 'xlsx'].includes(ext || '')) return <FileSpreadsheet size={20} className="text-green-400" />;
    if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(ext || '')) return <ImageIcon size={20} className="text-indigo-400" />;
    return <FileText size={20} className="text-gray-400" />;
  };

  const handleUpload = async () => {
    // Determine what to upload based on current inputs, regardless of active tab
    const hasFiles = files.length > 0;
    const hasUrl = !!urlInput.trim();
    
    if (!hasFiles && !hasUrl) return;

    setIsUploading(true);
    setUploadProgress(10);
    setUploadStatusText("Starting...");
    
    let allSources: any[] = [];
    let filePromise = Promise.resolve();
    let urlPromise = Promise.resolve();

    // 1. Files
    if (hasFiles) {
      filePromise = (async () => {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        try {
          const progressInterval = setInterval(() => {
            setUploadProgress(prev => Math.min(prev + 15, 90));
          }, 500);

          const res = await fetch(`${backendUrl}/api/sources/upload`, {
            method: 'POST',
            body: formData
          });
          clearInterval(progressInterval);
          
          const data = await res.json();
          if (data.success) {
            allSources = [...allSources, ...data.sources];
          } else {
            alert('File upload failed: ' + (data.detail || 'Unknown error'));
          }
        } catch (err) {
          alert('File upload failed');
        }
      })();
    }

    // 2. URL (Streaming)
    if (hasUrl) {
      urlPromise = (async () => {
        try {
          const res = await fetch(`${backendUrl}/api/sources/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlInput.trim() })
          });
          if (!res.body) throw new Error("No response body");
          
          const reader = res.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let done = false;
          
          while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
              const chunk = decoder.decode(value, { stream: true });
              const lines = chunk.split('\n').filter(Boolean);
              for (const line of lines) {
                try {
                  const data = JSON.parse(line);
                  if (data.status) {
                    setUploadStatusText(data.status);
                    setUploadProgress(prev => Math.min(prev + 15, 90));
                  }
                  if (data.success !== undefined) {
                    if (data.success) {
                      allSources.push(data.source);
                    } else {
                      alert('Failed to process URL: ' + (data.detail || 'Unknown error'));
                    }
                  }
                } catch (e) {}
              }
            }
          }
        } catch (err) {
          alert('Failed to process URL');
        }
      })();
    }

    await Promise.all([filePromise, urlPromise]);
    
    setUploadProgress(100);
    setUploadStatusText("Complete!");
    
    if (allSources.length > 0) {
      setTimeout(() => {
        onUploadSuccess(allSources);
        resetAndClose();
      }, 500);
    } else {
      setIsUploading(false);
    }
  };

  const resetAndClose = () => {
    setFiles([]);
    setUrlInput('');
    setIsUploading(false);
    setUploadProgress(0);
    setUploadStatusText("Processing...");
    onClose();
  };

  if (!isOpen) return null;

  // The button is disabled if uploading, OR if BOTH inputs are empty
  const isUploadDisabled = isUploading || (files.length === 0 && !urlInput.trim());

  return (
    <AnimatePresence>
      <div className="modal-overlay">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="modal-content"
        >
          <div className="modal-header">
            <h2>Add Sources</h2>
            <button onClick={resetAndClose} className="icon-btn" disabled={isUploading}><X size={20} /></button>
          </div>

          <div className="modal-tabs">
            <button className={`tab ${activeTab === 'files' ? 'active' : ''}`} onClick={() => setActiveTab('files')}>Files</button>
            <button className={`tab ${activeTab === 'url' ? 'active' : ''}`} onClick={() => setActiveTab('url')}>URL / YouTube</button>
          </div>

          <div className="modal-body">
            {activeTab === 'files' ? (
              <>
                <div 
                  {...getRootProps()} 
                  className={`dropzone ${isDragActive ? 'active' : ''} ${isDragReject ? 'reject' : ''}`}
                >
                  <input {...getInputProps()} />
                  <UploadCloud size={48} className="dropzone-icon" />
                  <h3>Drag & drop files here, or click to browse</h3>
                  <p>Supports PDF, DOCX, TXT, CSV, XLSX, PPTX, MD, HTML, JSON, Audio, Images</p>
                </div>

                {files.length > 0 && (
                  <div className="file-list">
                    <h4>Selected Files ({files.length})</h4>
                    <div className="file-items">
                      {files.map((file, idx) => (
                        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} key={`${file.name}-${idx}`} className="file-item">
                          <div className="file-info">
                            {getFileIcon(file.name)}
                            <span className="file-name" title={file.name}>{file.name}</span>
                            <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                          </div>
                          <button className="icon-btn danger" onClick={() => removeFile(idx)} disabled={isUploading}>
                            <Trash2 size={16} />
                          </button>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="url-input-container">
                <div className="url-icon-wrapper" style={{ display: 'flex', gap: '12px', justifyContent: 'center', alignItems: 'center' }}>
                  <Link2 size={24} />
                  <PlaySquare size={24} color="#ff0000" />
                </div>
                <h3>Add a Web Link or YouTube Video</h3>
                <p>Paste a URL to automatically extract its text content or transcript.</p>
                <input 
                  type="text" 
                  className="url-input" 
                  placeholder="https://example.com or https://youtube.com/watch?v=..." 
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  disabled={isUploading}
                />
              </div>
            )}
          </div>

          <div className="modal-footer">
            {isUploading && (
              <div className="upload-progress-container">
                <div className="progress-bar">
                  <motion.div 
                    className="progress-fill" 
                    initial={{ width: 0 }} 
                    animate={{ width: `${uploadProgress}%` }} 
                  />
                </div>
                <span className="progress-text">
                  {uploadProgress === 100 ? <CheckCircle size={14} color="#4ade80" /> : <Loader2 size={14} className="animate-spin" />}
                  {uploadProgress === 100 ? "Complete!" : uploadStatusText}
                </span>
              </div>
            )}
            <div style={{flex: 1}}></div>
            <button className="btn-cancel" onClick={resetAndClose} disabled={isUploading}>Cancel</button>
            <button 
              className="btn-primary" 
              onClick={handleUpload} 
              disabled={isUploadDisabled}
            >
              {isUploading ? 'Processing...' : 'Add Sources'}
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
