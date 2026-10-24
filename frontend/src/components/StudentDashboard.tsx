import React, { useState, useEffect } from 'react';
import {
  FileText, Headphones, ArrowRight, LogOut,
  Layers, Plus, CheckCircle, RefreshCw,
  Edit3, Trash2, Loader2, AlertCircle, User
} from 'lucide-react';
import { aiFetch } from '../config/api';
import {
  isSupabaseConfigured,
  fetchNotebooksFromSupabase,
  createNotebookInSupabase,
  updateNotebookInSupabase,
  deleteNotebookFromSupabase
} from '../config/supabase';
import './StudentDashboard.css';

interface StudentDashboardProps {
  user: any;
  onOpenStudio?: () => void;
  onOpen?: () => void;
  onOpenNotebook?: (notebook: any) => void;
  onLogout: () => void;
  onOpenProfile?: () => void;
}

export const StudentDashboard: React.FC<StudentDashboardProps> = ({
  user,
  onOpenStudio,
  onOpen,
  onOpenNotebook,
  onLogout,
  onOpenProfile
}) => {
  const userName = user?.full_name || user?.name || 'Rohan Sharma';
  const userCollege = user?.college_name || user?.college || 'DKTE Society\'s Textile & Engineering Institute';
  const userDegree = user?.degree_program === 'Other' ? (user?.custom_degree || 'Student') : (user?.degree_program || 'B.Tech / B.E.');
  const userBranch = (user?.branch === 'Other Branch / Department' || user?.branch === 'Other')
    ? (user?.custom_branch || 'General')
    : (user?.branch ? user.branch.replace(/\s*\(.*?\)\s*/g, '') : 'Computer Science');
  const userAvatarUrl = user?.avatar_url || null;
  const avatarInitial = (userName.trim() ? userName.trim().charAt(0) : 'S').toUpperCase();

  const [stats, setStats] = useState({
    activeNotebooks: 0,
    uploadedSources: 0,
    audioSynthesis: 0,
    solvedPYQs: 0,
  });
  const [notebooks, setNotebooks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creatingNotebook, setCreatingNotebook] = useState(false);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      let fetchedNotebooks: any[] = [];

      // 1. If Supabase is configured and user is authenticated, query Supabase directly
      if (isSupabaseConfigured() && user?.id) {
        try {
          const directData = await fetchNotebooksFromSupabase(user.id);
          if (Array.isArray(directData)) {
            fetchedNotebooks = directData;
          }
        } catch (sbErr: any) {
          console.warn('[StudentDashboard] Supabase direct query note:', sbErr.message);
        }
      }

      // 2. Fallback to authenticated backend /api/notebooks endpoint
      if (fetchedNotebooks.length === 0) {
        try {
          const aiNbRes = await aiFetch('/api/notebooks');
          if (aiNbRes.ok) {
            const nbData = await aiNbRes.json();
            if (nbData.success && Array.isArray(nbData.notebooks)) {
              fetchedNotebooks = nbData.notebooks;
            }
          }
        } catch (beErr: any) {
          console.warn('[StudentDashboard] Backend notebook fetch note:', beErr.message);
        }
      }

      setNotebooks(fetchedNotebooks);
      setStats(prev => ({
        ...prev,
        activeNotebooks: fetchedNotebooks.length,
        uploadedSources: fetchedNotebooks.reduce((acc, nb) => acc + (nb.sources_count || nb.sourcesCount || 0), 0)
      }));

    } catch (err: any) {
      console.error('[StudentDashboard] Error fetching dashboard metrics:', err);
      setError('Unable to load notebooks. Please verify your connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateNewNotebook = async () => {
    if (creatingNotebook) return;
    setCreatingNotebook(true);
    setError(null);
    try {
      let createdNb: any = null;

      // 1. Create directly in Supabase if configured
      if (isSupabaseConfigured() && user?.id) {
        try {
          createdNb = await createNotebookInSupabase(user.id, 'Untitled Notebook', 'CS101');
        } catch (sbErr: any) {
          console.warn('[StudentDashboard] Supabase create error, falling back to API:', sbErr.message);
        }
      }

      // 2. Fallback to backend API
      if (!createdNb) {
        const res = await aiFetch('/api/notebooks', {
          method: 'POST',
          body: JSON.stringify({
            title: 'Untitled Notebook',
            code: 'CS101',
            content: ''
          })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.success && data.notebook) {
          createdNb = data.notebook;
        }
      }

      if (createdNb) {
        setNotebooks(prev => [createdNb, ...prev]);
        setStats(prev => ({ ...prev, activeNotebooks: prev.activeNotebooks + 1 }));
        if (onOpenNotebook) {
          onOpenNotebook(createdNb);
        } else if (onOpenStudio) {
          onOpenStudio();
        } else if (onOpen) {
          onOpen();
        }
      } else {
        throw new Error('Failed to create notebook record');
      }
    } catch (err: any) {
      console.error('[StudentDashboard] Create notebook error:', err);
      setError('Failed to create notebook. Please try again.');
    } finally {
      setCreatingNotebook(false);
    }
  };

  const handleRenameNotebook = async (nbId: string, currentTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newTitle = window.prompt('Enter new notebook title:', currentTitle);
    if (!newTitle || newTitle.trim() === currentTitle) return;

    try {
      if (isSupabaseConfigured() && user?.id) {
        try {
          await updateNotebookInSupabase(nbId, user.id, { title: newTitle.trim() });
        } catch (sbErr: any) {
          console.warn('[StudentDashboard] Supabase rename fallback to API:', sbErr.message);
          await aiFetch(`/api/notebooks/${nbId}`, {
            method: 'PUT',
            body: JSON.stringify({ title: newTitle.trim() })
          });
        }
      } else {
        await aiFetch(`/api/notebooks/${nbId}`, {
          method: 'PUT',
          body: JSON.stringify({ title: newTitle.trim() })
        });
      }

      setNotebooks(prev => prev.map(nb => (nb.id === nbId || nb._id === nbId ? { ...nb, title: newTitle.trim() } : nb)));
    } catch (err) {
      console.error('[StudentDashboard] Rename error:', err);
    }
  };

  const handleDeleteNotebook = async (nbId: string, title: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete "${title}"? All saved notes and attachments will be removed.`)) return;

    try {
      if (isSupabaseConfigured() && user?.id) {
        try {
          await deleteNotebookFromSupabase(nbId, user.id);
        } catch (sbErr: any) {
          console.warn('[StudentDashboard] Supabase delete fallback to API:', sbErr.message);
          await aiFetch(`/api/notebooks/${nbId}`, { method: 'DELETE' });
        }
      } else {
        await aiFetch(`/api/notebooks/${nbId}`, { method: 'DELETE' });
      }

      setNotebooks(prev => prev.filter(nb => nb.id !== nbId && nb._id !== nbId));
      setStats(prev => ({ ...prev, activeNotebooks: Math.max(0, prev.activeNotebooks - 1) }));
    } catch (err) {
      console.error('[StudentDashboard] Delete error:', err);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, [user?.id]);

  return (
    <div className="sd-container">
      {/* ── Welcome Header ─────────────────────────────────────────── */}
      <header className="sd-header">
        <div
          className="sd-user-info"
          onClick={onOpenProfile}
          style={{ cursor: onOpenProfile ? 'pointer' : 'default' }}
          title={onOpenProfile ? 'View and edit your student profile' : undefined}
        >
          <div className="sd-avatar">
            {userAvatarUrl ? (
              <img src={userAvatarUrl} alt={userName} className="sd-avatar-img" />
            ) : (
              <span>{avatarInitial}</span>
            )}
          </div>
          <div>
            <h1 className="sd-welcome-title">Welcome back, {userName}</h1>
            <p className="sd-welcome-sub">
              {userDegree} • {userBranch} • {userCollege}
            </p>
          </div>
        </div>

        <div className="sd-header-actions">
          {onOpenProfile && (
            <button
              className="sd-profile-btn"
              onClick={onOpenProfile}
              title="Open full student profile"
            >
              <User size={15} />
              My Profile
            </button>
          )}

          <button className="sd-logout-btn" onClick={onLogout} title="Sign out of StudySnap">
            <LogOut size={15} />
            Log Out
          </button>
        </div>
      </header>

      {/* ── BIG CHATBOT LAUNCH CARD ─────────────────────────────────── */}
      {/* <section className="sd-chatbot-card">
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
      </section> */}

      {/* ── Quick Stats Grid ───────────────────────────────────────── */}
      <section className="sd-stats-grid">
        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <Layers size={22} />
          </div>
          <div>
            <div className="sd-stat-value">{stats.activeNotebooks}</div>
            <div className="sd-stat-label">Active Notebooks</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <FileText size={22} />
          </div>
          <div>
            <div className="sd-stat-value">{stats.uploadedSources}</div>
            <div className="sd-stat-label">Uploaded Sources</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <Headphones size={22} />
          </div>
          <div>
            <div className="sd-stat-value">{stats.audioSynthesis}</div>
            <div className="sd-stat-label">Audio Synthesis</div>
          </div>
        </div>

        <div className="sd-stat-card">
          <div className="sd-stat-icon">
            <CheckCircle size={22} />
          </div>
          <div>
            <div className="sd-stat-value">{stats.solvedPYQs}</div>
            <div className="sd-stat-label">Solved Exam PYQs</div>
          </div>
        </div>
      </section>

      {/* ── My Notebooks Section ───────────────────────────────────── */}
      <section>
        <div className="sd-section-title">
          <span>My Active Study Notebooks</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={fetchDashboardData}
              title="Refresh dashboard metrics"
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#94a3b8',
                padding: '6px 14px',
                borderRadius: '9999px',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <RefreshCw
                size={14}
                style={{
                  animation: loading ? 'spin 1s linear infinite' : 'none'
                }}
              />
              <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
            </button>
            <button
              onClick={handleCreateNewNotebook}
              disabled={creatingNotebook}
              style={{
                background: 'rgba(245, 158, 11, 0.15)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                color: '#FBBF24',
                padding: '6px 14px',
                borderRadius: '9999px',
                fontSize: '0.85rem',
                fontWeight: 700,
                cursor: creatingNotebook ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              {creatingNotebook ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Plus size={16} />
                  New Notebook
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            padding: '12px 16px',
            borderRadius: '12px',
            margin: '16px 0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
            <button
              onClick={fetchDashboardData}
              style={{
                background: '#ef4444',
                color: '#fff',
                border: 'none',
                padding: '4px 12px',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '0.8rem',
                fontWeight: 600
              }}
            >
              Retry
            </button>
          </div>
        )}

        {loading && notebooks.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '60px 24px', color: '#94a3b8' }}>
            <Loader2 size={36} className="animate-spin" style={{ margin: '0 auto 12px auto', color: '#f59e0b' }} />
            <p style={{ fontSize: '0.95rem' }}>Loading your study notebooks from Supabase...</p>
          </div>
        ) : notebooks.length === 0 ? (
          <div
            style={{
              textAlign: 'center',
              padding: '48px 24px',
              background: 'rgba(255, 255, 255, 0.02)',
              borderRadius: '16px',
              border: '1px dashed rgba(255, 255, 255, 0.1)',
              marginTop: '16px'
            }}
          >
            <Layers size={42} style={{ color: '#F59E0B', marginBottom: '12px', opacity: 0.8 }} />
            <h3 style={{ fontSize: '1.1rem', color: '#F3F4F6', marginBottom: '6px' }}>
              No Active Notebooks Yet
            </h3>
            <p style={{ color: '#9CA3AF', fontSize: '0.875rem', maxWidth: '420px', margin: '0 auto 18px auto' }}>
              Create your first study notebook to upload lecture documents, chat with sources via RAG, and generate mind maps.
            </p>
            <button
              onClick={handleCreateNewNotebook}
              disabled={creatingNotebook}
              style={{
                background: '#F59E0B',
                color: '#111827',
                border: 'none',
                padding: '8px 18px',
                borderRadius: '9999px',
                fontWeight: 700,
                fontSize: '0.875rem',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Plus size={16} />
              Create Your First Notebook
            </button>
          </div>
        ) : (
          <div className="sd-notebooks-grid">
            {notebooks.map(nb => (
              <div
                key={nb.id || nb._id}
                className="sd-notebook-card"
                onClick={() => (onOpenNotebook ? onOpenNotebook(nb) : (onOpenStudio ? onOpenStudio() : onOpen?.()))}
                style={{ cursor: 'pointer' }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div>
                    <span className="sd-notebook-tag">{nb.code || 'CS101'}</span>
                    <h3 className="sd-notebook-title">{nb.title}</h3>
                    <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                      {nb.updated_at ? `Updated ${new Date(nb.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}` : (nb.lastUpdated || 'Recently edited')}
                    </p>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <button
                      onClick={(e) => handleRenameNotebook(nb.id || nb._id, nb.title, e)}
                      title="Rename Notebook"
                      className="icon-btn"
                      style={{ padding: '4px', color: '#94a3b8' }}
                    >
                      <Edit3 size={15} />
                    </button>
                    <button
                      onClick={(e) => handleDeleteNotebook(nb.id || nb._id, nb.title, e)}
                      title="Delete Notebook"
                      className="icon-btn"
                      style={{ padding: '4px', color: '#f87171' }}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </div>

                <div className="sd-notebook-meta" style={{ marginTop: 'auto' }}>
                  <span>{nb.sources_count || nb.sourcesCount || 0} Sources</span>
                  <span>•</span>
                  <span>{nb.notes_count || nb.notesCount || 0} AI Notes</span>
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
        )}
      </section>
    </div>
  );
};

export default StudentDashboard;
