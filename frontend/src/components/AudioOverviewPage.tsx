import React, { useState } from 'react';
import { ArrowLeft, Activity, Mic2, FileText, CheckCircle2, MessageSquare, ChevronDown, Loader2 } from 'lucide-react';

interface AudioOverviewPageProps {
  backendUrl: string;
  selectedSourceIds: Set<string>;
  responseLanguage: string;
  onBack: () => void;
}

const formats = [
  { id: 'deep_dive', name: 'Deep Dive', desc: 'A lively conversation between two hosts, unpacking and connecting topics.', icon: Mic2 },
  { id: 'brief', name: 'Brief', desc: 'A bite-sized overview to help you grasp core ideas quickly.', icon: FileText },
  { id: 'critique', name: 'Critique', desc: 'An expert review offering constructive feedback.', icon: CheckCircle2 },
  { id: 'debate', name: 'Debate', desc: 'A thoughtful debate illuminating different perspectives.', icon: MessageSquare },
];

export const AudioOverviewPage: React.FC<AudioOverviewPageProps> = ({ backendUrl, selectedSourceIds, responseLanguage, onBack }) => {
  const [selectedFormat, setSelectedFormat] = useState('deep_dive');
  const [language, setLanguage] = useState(responseLanguage);
  const [length, setLength] = useState('Default');
  const [instructions, setInstructions] = useState('');
  
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateAudio = async () => {
    if (selectedSourceIds.size === 0) {
      setError("Please select at least one source document in the Studio to generate an audio overview.");
      return;
    }
    setLoading(true);
    setError(null);
    setAudioUrl(null);
    try {
      const res = await fetch(`${backendUrl}/api/audio-overview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: language
          // Note: In a complete implementation, length and instructions would also be passed
        }),
      });
      const data = await res.json();
      if (data.success && data.audio_url) {
        setAudioUrl(data.audio_url);
      } else {
        setError("Failed to generate audio overview.");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="feature-page-container" style={{ padding: '24px', maxWidth: '800px', margin: '0 auto', height: '100%', overflowY: 'auto' }}>
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
          marginBottom: '24px',
          transition: 'all 0.2s'
        }}
      >
        <ArrowLeft size={16} />
        Back to Studio
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
        <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: 'var(--accent-color)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Activity size={28} style={{ color: '#000' }} />
        </div>
        <div>
          <h1 style={{ fontSize: '28px', fontWeight: 600, margin: '0 0 4px 0', color: 'var(--text-primary)' }}>Audio Overview</h1>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '15px' }}>Turn your study materials into an engaging podcast.</p>
        </div>
      </div>

      {!audioUrl && !loading && (
        <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Format Selection */}
          <div>
            <label style={{ display: 'block', fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '16px' }}>Podcast Format</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '16px' }}>
              {formats.map(f => (
                <div 
                  key={f.id}
                  onClick={() => setSelectedFormat(f.id)}
                  style={{ 
                    padding: '20px', 
                    borderRadius: '12px', 
                    border: selectedFormat === f.id ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                    backgroundColor: selectedFormat === f.id ? 'rgba(168, 199, 250, 0.05)' : 'var(--bg-button)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <f.icon size={20} color={selectedFormat === f.id ? 'var(--accent-color)' : 'var(--text-secondary)'} />
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '16px' }}>{f.name}</span>
                  </div>
                  <p style={{ fontSize: '14px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>{f.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {/* Language */}
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>Language</label>
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
                    fontSize: '15px',
                    appearance: 'none',
                    cursor: 'pointer',
                    outline: 'none'
                  }}
                >
                  <option value="English">English</option>
                  <option value="Hinglish">Hinglish</option>
                  <option value="Manglish" disabled>Manglish — Coming Soon</option>
                  <option value="Hindi" disabled>Hindi — Coming Soon</option>
                  <option value="Marathi" disabled>Marathi — Coming Soon</option>
                </select>
                <ChevronDown size={18} style={{ position: 'absolute', right: '16px', top: '12px', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
              </div>
            </div>

            {/* Length */}
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>Length</label>
              <div style={{ display: 'flex', backgroundColor: 'var(--bg-button)', borderRadius: '8px', padding: '4px', border: '1px solid var(--border-color)' }}>
                {['Short', 'Default', 'Long'].map(l => (
                  <button
                    key={l}
                    onClick={() => setLength(l)}
                    style={{
                      flex: 1,
                      padding: '10px 0',
                      fontSize: '14px',
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
            <label style={{ display: 'block', fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>Focus area (Optional)</label>
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
                fontSize: '15px',
                resize: 'vertical',
                outline: 'none'
              }}
            />
          </div>

          <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end' }}>
            <button 
              onClick={generateAudio}
              style={{ padding: '12px 32px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '16px', fontWeight: 600, cursor: 'pointer' }}
            >
              Generate Podcast
            </button>
          </div>
          {error && <p style={{ color: '#ef4444', textAlign: 'right', margin: 0 }}>{error}</p>}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', marginTop: '80px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <Loader2 size={40} className="animate-spin" style={{ color: 'var(--accent-color)' }} />
          <h3 style={{ fontSize: '18px', fontWeight: 500 }}>Writing script and synthesizing voices...</h3>
          <p style={{ color: 'var(--text-secondary)' }}>This usually takes 10–15 seconds.</p>
        </div>
      )}

      {audioUrl && !loading && (
        <div style={{ backgroundColor: 'var(--bg-panel)', padding: '48px 32px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '32px' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(168, 199, 250, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
             <Mic2 size={40} style={{ color: '#a8c7fa' }} />
          </div>
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 8px 0' }}>Your Audio Overview is Ready</h2>
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Listen to the generated podcast below.</p>
          </div>
          
          <div style={{ width: '100%', maxWidth: '600px', backgroundColor: 'var(--bg-button)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-color)' }}>
            <audio 
              controls 
              src={audioUrl} 
              style={{ width: '100%', outline: 'none' }} 
              autoPlay
            />
          </div>

          <button 
            onClick={() => setAudioUrl(null)}
            style={{ padding: '12px 24px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '15px', fontWeight: 600, cursor: 'pointer', marginTop: '16px' }}
          >
            Create Another
          </button>
        </div>
      )}
    </div>
  );
};
