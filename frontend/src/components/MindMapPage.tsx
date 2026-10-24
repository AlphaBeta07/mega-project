import React, { useState } from 'react';
import { ArrowLeft, Loader2, Network } from 'lucide-react';
import MindMapRenderer from '../MindMapRenderer';

interface MindMapPageProps {
  backendUrl: string;
  selectedSourceIds: Set<string>;
  responseLanguage: string;
  onBack: () => void;
  onNodeClick: (nodeId: string, label: string, context: string) => void;
}

export const MindMapPage: React.FC<MindMapPageProps> = ({ backendUrl, selectedSourceIds, responseLanguage, onBack, onNodeClick }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');

  const generateMindMap = async () => {
    if (selectedSourceIds.size === 0) {
      setError("Please select at least one source document in the Studio to generate a Mind Map.");
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch(`${backendUrl}/api/mindmap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          custom_prompt: customPrompt,
          response_language: responseLanguage
        }),
      });
      const resData = await res.json();
      if (resData.success && resData.mind_map) {
        setData(resData.mind_map);
      } else {
        setError("Failed to generate Mind Map.");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="feature-page-container" style={{ padding: 'clamp(12px, 3vw, 24px)', display: 'flex', flexDirection: 'column', height: '100%', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', marginBottom: '20px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <button
            onClick={onBack}
            style={{
              background: 'var(--surface-secondary, rgba(255, 255, 255, 0.08))',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
              padding: '6px 16px',
              borderRadius: '9999px',
              fontSize: '0.85rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s'
            }}
          >
            <ArrowLeft size={16} />
            Back to Studio
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
             <Network size={20} style={{ color: '#e5c9b3' }} />
             <h2 style={{ margin: 0, fontSize: 'clamp(16px, 4vw, 18px)', fontWeight: 600 }}>Mind Map Explorer</h2>
          </div>
        </div>

        {data && (
          <button 
            onClick={() => setData(null)}
            style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '14px', fontWeight: 500, cursor: 'pointer' }}
          >
            Create New Map
          </button>
        )}
      </div>

      {!data && !loading && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(229, 201, 179, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
             <Network size={40} style={{ color: '#e5c9b3' }} />
          </div>
          <h2 style={{ fontSize: '24px', fontWeight: 600, marginBottom: '12px' }}>Generate a Mind Map</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', textAlign: 'center', maxWidth: '400px' }}>
            Map out concepts and their relationships from your study materials.
          </p>
          
          <div style={{ width: '100%', maxWidth: '500px', marginBottom: '24px' }}>
             <label style={{ display: 'block', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '8px' }}>Custom Instructions (Optional)</label>
              <textarea 
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="E.g., Map out the pros and cons, Focus on the chronological events, Create a Tension Map for the core debate."
                style={{ width: '100%', minHeight: '100px', padding: '16px', borderRadius: '12px', backgroundColor: 'var(--bg-button)', border: '1px solid var(--border-color)', color: 'var(--text-primary)', fontSize: '14px', resize: 'vertical' }}
              />
          </div>

          <button 
            onClick={generateMindMap}
            style={{ padding: '12px 24px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '15px', fontWeight: 600, cursor: 'pointer' }}
          >
            Start Generating
          </button>
          {error && <p style={{ color: '#ef4444', marginTop: '16px' }}>{error}</p>}
        </div>
      )}

      {loading && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
          <Loader2 size={40} className="animate-spin" style={{ color: 'var(--accent-color)' }} />
          <h3 style={{ fontSize: '18px', fontWeight: 500 }}>Connecting concepts and building nodes...</h3>
        </div>
      )}

      {data && !loading && (
        <div style={{ flex: 1, minHeight: 0, borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
          <MindMapRenderer data={data} onNodeClick={onNodeClick} />
        </div>
      )}
    </div>
  );
};
