import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Activity, Mic2, FileText, CheckCircle2, MessageSquare, ChevronDown } from 'lucide-react';

interface AudioOverviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (format: string, language: string, length: string, instructions: string) => void;
}

const formats = [
  { id: 'deep_dive', name: 'Deep Dive', desc: 'A lively conversation between two hosts, unpacking and connecting topics.', icon: Mic2 },
  { id: 'brief', name: 'Brief', desc: 'A bite-sized overview to help you grasp core ideas quickly.', icon: FileText },
  { id: 'critique', name: 'Critique', desc: 'An expert review offering constructive feedback.', icon: CheckCircle2 },
  { id: 'debate', name: 'Debate', desc: 'A thoughtful debate illuminating different perspectives.', icon: MessageSquare },
];

export default function AudioOverviewModal({ isOpen, onClose, onGenerate }: AudioOverviewModalProps) {
  const [selectedFormat, setSelectedFormat] = useState('deep_dive');
  const [language, setLanguage] = useState('English');
  const [length, setLength] = useState('Default');
  const [instructions, setInstructions] = useState('');

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="modal-overlay">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="modal-content"
          style={{ maxWidth: '600px', padding: 0, overflow: 'hidden' }}
        >
          {/* Header */}
          <div style={{ padding: '20px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ background: 'var(--accent-color)', color: '#000', padding: '6px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Activity size={18} />
              </div>
              <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Customize Audio Overview</h2>
            </div>
            <button onClick={onClose} className="icon-btn" style={{ background: 'transparent' }}><X size={20} /></button>
          </div>

          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', maxHeight: '70vh', overflowY: 'auto' }}>
            
            {/* Format Selection */}
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Format</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                {formats.map(f => (
                  <div 
                    key={f.id}
                    onClick={() => setSelectedFormat(f.id)}
                    style={{ 
                      padding: '16px', 
                      borderRadius: '12px', 
                      border: selectedFormat === f.id ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                      backgroundColor: selectedFormat === f.id ? 'rgba(168, 199, 250, 0.05)' : 'transparent',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <f.icon size={18} color={selectedFormat === f.id ? 'var(--accent-color)' : 'var(--text-secondary)'} />
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '15px' }}>{f.name}</span>
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>{f.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '24px' }}>
              {/* Language */}
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Language</label>
                <div style={{ position: 'relative' }}>
                  <select 
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    style={{ 
                      width: '100%', 
                      padding: '12px 16px', 
                      borderRadius: '8px', 
                      backgroundColor: 'var(--bg-button)', 
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-primary)',
                      fontSize: '14px',
                      appearance: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    <option>English</option>
                    <option>Spanish</option>
                    <option>French</option>
                    <option>German</option>
                    <option>Hindi</option>
                  </select>
                  <ChevronDown size={16} style={{ position: 'absolute', right: '16px', top: '14px', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                </div>
              </div>

              {/* Length */}
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Length</label>
                <div style={{ display: 'flex', backgroundColor: 'var(--bg-button)', borderRadius: '8px', padding: '4px', border: '1px solid var(--border-color)' }}>
                  {['Short', 'Default', 'Long'].map(l => (
                    <button
                      key={l}
                      onClick={() => setLength(l)}
                      style={{
                        flex: 1,
                        padding: '8px 0',
                        fontSize: '13px',
                        fontWeight: 500,
                        borderRadius: '6px',
                        backgroundColor: length === l ? 'var(--bg-secondary)' : 'transparent',
                        color: length === l ? 'var(--text-primary)' : 'var(--text-secondary)',
                        border: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        boxShadow: length === l ? '0 1px 3px rgba(0,0,0,0.2)' : 'none'
                      }}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Focus Area */}
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Focus area (Optional)</label>
              <textarea 
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="What should the AI hosts focus on in this episode?"
                style={{
                  width: '100%',
                  minHeight: '100px',
                  padding: '16px',
                  borderRadius: '12px',
                  backgroundColor: 'var(--bg-button)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  resize: 'vertical'
                }}
              />
            </div>

          </div>

          {/* Footer */}
          <div style={{ padding: '20px 24px', display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}>
            <button 
              onClick={onClose} 
              style={{ padding: '10px 20px', borderRadius: '20px', border: 'none', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '14px', fontWeight: 500, cursor: 'pointer' }}
            >
              Cancel
            </button>
            <button 
              onClick={() => {
                onGenerate(selectedFormat, language, length, instructions);
                onClose();
              }} 
              style={{ padding: '10px 24px', borderRadius: '20px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '14px', fontWeight: 600, cursor: 'pointer' }}
            >
              Generate
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
