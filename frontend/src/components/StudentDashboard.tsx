import React from 'react';
import {
  Sparkles, FileText, Headphones, ArrowRight, LogOut,
  BrainCircuit, Layers, Plus, CheckCircle
} from 'lucide-react';
import './StudentDashboard.css';

interface StudentDashboardProps {
  user: { name: string; email: string; college: string } | null;
  onOpenStudio: () => void;
  onLogout: () => void;
}

export const StudentDashboard: React.FC<StudentDashboardProps> = ({
  user,
  onOpenStudio,
  onLogout
}) => {
  const userName = user?.name || 'Rohan Sharma';
  const userCollege = user?.college || 'IIT Bombay';

  const mockNotebooks = [
    {
      id: 'nb-1',
      code: 'CS302',
      title: 'Operating Systems — Process Scheduling & Deadlocks',
      sourcesCount: 4,
      notesCount: 12,
      lastUpdated: '2 hours ago'
    },
    {
      id: 'nb-2',
      code: 'CS201',
      title: 'Data Structures — AVL Tree Rotations & Red-Black Trees',
      sourcesCount: 3,
      notesCount: 8,
      lastUpdated: 'Yesterday'
    },
    {
      id: 'nb-3',
      code: 'EC401',
      title: 'Digital Signal Processing — FFT Decimation Algorithm',
      sourcesCount: 2,
      notesCount: 5,
      lastUpdated: '3 days ago'
    },
    {
      id: 'nb-4',
      code: 'CS504',
      title: 'Database Systems — 3NF & BCNF Normalization PYQs',
      sourcesCount: 5,
      notesCount: 15,
      lastUpdated: '4 days ago'
    }
  ];

  return (
    <div className="sd-container">
      {/* ── Welcome Header ─────────────────────────────────────────── */}
      <header className="sd-header">
        <div className="sd-user-info">
          <div className="sd-avatar">
            {userName.charAt(0)}
          </div>
          <div>
            <h1 className="sd-welcome-title">Welcome back, {userName} 👋</h1>
            <p className="sd-welcome-sub">Computer Science • {userCollege}</p>
          </div>
        </div>

        <button className="sd-logout-btn" onClick={onLogout}>
          <LogOut size={16} />
          Log Out
        </button>
      </header>

      {/* ── BIG CHATBOT LAUNCH CARD ─────────────────────────────────── */}
      <section className="sd-chatbot-card">
        <div className="sd-chatbot-info">
          <div className="sd-chatbot-badge">
            <Sparkles size={14} />
            POWERED BY LM STUDIO & RAG AI
          </div>
          <h2 className="sd-chatbot-title">
            AI Study Assistant & RAG Chatbot
          </h2>
          <p className="sd-chatbot-desc">
            Upload your lecture recordings, PDFs, slides, or YouTube links. Chat directly with your sources, generate 2-min Audio Overviews, interactive Mind Maps, Infographics, and solved PYQ exam predictions.
          </p>
        </div>

        <button className="sd-chatbot-launch-btn" onClick={onOpenStudio}>
          <BrainCircuit size={22} />
          <span>Launch AI Chatbot Studio</span>
          <ArrowRight size={18} />
        </button>
      </section>

      {/* ── Quick Stats Grid ───────────────────────────────────────── */}
      <section className="sd-stats-grid">
        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <Layers size={22} />
          </div>
          <div>
            <div className="sd-stat-value">4</div>
            <div className="sd-stat-label">Active Notebooks</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <FileText size={22} />
          </div>
          <div>
            <div className="sd-stat-value">14</div>
            <div className="sd-stat-label">Uploaded Sources</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <Headphones size={22} />
          </div>
          <div>
            <div className="sd-stat-value">3</div>
            <div className="sd-stat-label">Audio Synthesis</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <CheckCircle size={22} />
          </div>
          <div>
            <div className="sd-stat-value">28</div>
            <div className="sd-stat-label">Solved Exam PYQs</div>
          </div>
        </div>
      </section>

      {/* ── My Notebooks Section ───────────────────────────────────── */}
      <section>
        <div className="sd-section-title">
          <span>My Active Study Notebooks</span>
          <button
            onClick={onOpenStudio}
            style={{
              background: 'rgba(245, 158, 11, 0.15)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              color: '#FBBF24',
              padding: '6px 14px',
              borderRadius: '9999px',
              fontSize: '0.85rem',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Plus size={16} />
            Create New Notebook
          </button>
        </div>

        <div className="sd-notebooks-grid">
          {mockNotebooks.map(nb => (
            <div key={nb.id} className="sd-notebook-card" onClick={onOpenStudio}>
              <div>
                <span className="sd-notebook-tag">{nb.code}</span>
                <h3 className="sd-notebook-title">{nb.title}</h3>
              </div>

              <div className="sd-notebook-meta">
                <span>{nb.sourcesCount} Sources</span>
                <span>•</span>
                <span>{nb.notesCount} AI Notes</span>
                <span style={{ marginLeft: 'auto' }}>
                  <button className="sd-open-btn">
                    <span>Open</span>
                    <ArrowRight size={14} />
                  </button>
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default StudentDashboard;
