import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Network } from 'lucide-react';

interface MindMapModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (customPrompt: string) => void;
}

export default function MindMapModal({ isOpen, onClose, onGenerate }: MindMapModalProps) {
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
                <Network size={18} />
              </div>
              <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Create Mind Map</h2>
            </div>
            <button onClick={onClose} className="icon-btn" style={{ background: 'transparent' }}><X size={20} /></button>
          </div>

          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', maxHeight: '70vh', overflowY: 'auto' }}>
            {/* Custom Prompt */}
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px' }}>Custom Instructions (Optional)</label>
              <textarea 
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="E.g., Map out the pros and cons, Focus on the chronological events, Create a Tension Map for the core debate."
                style={{ width: '100%', minHeight: '100px', padding: '16px', borderRadius: '12px', backgroundColor: 'var(--bg-button)', border: '1px solid var(--border-color)', color: 'var(--text-primary)', fontSize: '14px', resize: 'vertical' }}
              />
            </div>
          </div>

          {/* Footer */}
          <div style={{ padding: '20px 24px', display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}>
            <button onClick={onClose} style={{ padding: '10px 20px', borderRadius: '20px', border: 'none', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '14px', fontWeight: 500, cursor: 'pointer' }}>Cancel</button>
            <button onClick={() => { onGenerate(customPrompt); onClose(); }} style={{ padding: '10px 24px', borderRadius: '20px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '14px', fontWeight: 600, cursor: 'pointer' }}>Generate</button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
