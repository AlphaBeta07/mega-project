import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, FileBarChart, Palette } from 'lucide-react';

interface InfographicModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (style: string, detailLevel: string, customPrompt: string) => void;
}

export default function InfographicModal({ isOpen, onClose, onGenerate }: InfographicModalProps) {
  const [style, setStyle] = useState('Bento Grid');
  const [detailLevel, setDetailLevel] = useState('Standard');
  const [customPrompt, setCustomPrompt] = useState('');

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
                <FileBarChart size={18} />
              </div>
              <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Create Infographic</h2>
            </div>
            <button onClick={onClose} className="icon-btn" style={{ background: 'transparent' }}><X size={20} /></button>
          </div>

          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', maxHeight: '70vh', overflowY: 'auto' }}>
            
            <div style={{ display: 'flex', gap: '24px' }}>
              {/* Style */}
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Visual Style</label>
                <div style={{ position: 'relative' }}>
                  <select 
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', backgroundColor: 'var(--bg-button)', border: '1px solid var(--border-color)', color: 'var(--text-primary)', fontSize: '14px', appearance: 'none', cursor: 'pointer' }}
                  >
                    <option>Bento Grid</option>
                    <option>Scientific</option>
                    <option>Professional</option>
                  </select>
                  <Palette size={16} style={{ position: 'absolute', right: '16px', top: '14px', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                </div>
              </div>

              {/* Detail Level */}
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Detail Level</label>
                <div style={{ display: 'flex', backgroundColor: 'var(--bg-button)', borderRadius: '8px', padding: '4px', border: '1px solid var(--border-color)' }}>
                  {['Concise', 'Standard', 'Detailed'].map(l => (
                    <button
                      key={l}
                      onClick={() => setDetailLevel(l)}
                      style={{ flex: 1, padding: '8px 0', fontSize: '13px', fontWeight: 500, borderRadius: '6px', backgroundColor: detailLevel === l ? 'var(--bg-secondary)' : 'transparent', color: detailLevel === l ? 'var(--text-primary)' : 'var(--text-secondary)', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', boxShadow: detailLevel === l ? '0 1px 3px rgba(0,0,0,0.2)' : 'none' }}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Custom Prompt */}
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Custom Instructions (Optional)</label>
              <textarea 
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="E.g., Highlight the main three statistics, focus on the timeline of events, use a minimalist approach."
                style={{ width: '100%', minHeight: '100px', padding: '16px', borderRadius: '12px', backgroundColor: 'var(--bg-button)', border: '1px solid var(--border-color)', color: 'var(--text-primary)', fontSize: '14px', resize: 'vertical' }}
              />
            </div>

          </div>

          {/* Footer */}
          <div style={{ padding: '20px 24px', display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}>
            <button onClick={onClose} style={{ padding: '10px 20px', borderRadius: '20px', border: 'none', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '14px', fontWeight: 500, cursor: 'pointer' }}>Cancel</button>
            <button onClick={() => { onGenerate(style, detailLevel, customPrompt); onClose(); }} style={{ padding: '10px 24px', borderRadius: '20px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '14px', fontWeight: 600, cursor: 'pointer' }}>Generate</button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
