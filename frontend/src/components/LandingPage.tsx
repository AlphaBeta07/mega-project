import React, { useState, useMemo, useEffect } from 'react';
import { Search, BookOpen, Plus, User, ArrowRight, Loader2, RefreshCw } from 'lucide-react';
import { MOCK_PUBLIC_NOTES } from '../data/mockNotes';
import type { PublicNote } from '../data/mockNotes';
import { API_GATEWAY_URL } from '../config/api';
import './LandingPage.css';

interface LandingPageProps {
  onCreateNotebook: () => void;
  onSelectNote: (note: PublicNote) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onCreateNotebook,
  onSelectNote,
}) => {
  // Live Data State
  const [notes, setNotes] = useState<PublicNote[]>(MOCK_PUBLIC_NOTES);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Filter States
  const [selectedCollege, setSelectedCollege] = useState('All Colleges');
  const [selectedBranch, setSelectedBranch] = useState('All Branches');
  const [selectedSemester, setSelectedSemester] = useState('All Semesters');
  const [selectedSubject, setSelectedSubject] = useState('All Subjects');
  const [selectedNoteType, setSelectedNoteType] = useState('All Types');
  const [selectedSortBy, setSelectedSortBy] = useState('Most Recent');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch Public Notes from Backend
  const fetchPublicNotes = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_GATEWAY_URL}/api/notes/public`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success && Array.isArray(data.data?.notes) && data.data.notes.length > 0) {
        const mappedNotes: PublicNote[] = data.data.notes.map((n: any) => ({
          id: n.note_id || n.id || n._id,
          code: n.code || 'GEN101',
          title: n.title || `${n.subject} — ${n.chapter}`,
          date: n.created_at ? new Date(n.created_at).toISOString().split('T')[0] : (n.date || '2026-03-10'),
          college: n.college || 'DKTE',
          branch: n.branch || 'Computer Science',
          semester: n.semester || 'Sem 5',
          subject: n.subject || 'General Study',
          noteType: (n.note_type || n.noteType || 'Lecture Notes') as PublicNote['noteType'],
          description: n.description || n.summary || '',
          author: n.author || 'StudySnap Student',
          likesCount: n.likes_count ?? n.likesCount ?? 0,
          readTime: n.read_time || n.readTime || '8 min read',
          fullContent: {
            overview: n.full_content?.overview || n.fullContent?.overview || n.summary || '',
            keyTopics: n.full_content?.key_topics || n.fullContent?.keyTopics || n.keywords || [],
            pyqs: n.full_content?.pyqs || n.fullContent?.pyqs || [],
            formulas: n.full_content?.formulas || n.fullContent?.formulas || [],
          }
        }));
        setNotes(mappedNotes);
      }
    } catch (err: any) {
      console.warn('[LandingPage] Backend notes fetch warning, using seed data:', err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPublicNotes();
  }, []);

  // Extract unique filter dropdown options dynamically from current notes
  const colleges = useMemo(() => ['All Colleges', ...Array.from(new Set(notes.map(n => n.college)))], [notes]);
  const branches = useMemo(() => ['All Branches', ...Array.from(new Set(notes.map(n => n.branch)))], [notes]);
  const semesters = useMemo(() => ['All Semesters', ...Array.from(new Set(notes.map(n => n.semester)))], [notes]);
  const subjects = useMemo(() => ['All Subjects', ...Array.from(new Set(notes.map(n => n.subject)))], [notes]);
  const noteTypes = useMemo(() => ['All Types', 'Lecture Notes', 'PYQ & Solutions', 'Formula Sheet', 'Lab Manual'], []);

  // Filter & Sort Logic
  const filteredNotes = useMemo(() => {
    return notes.filter(note => {
      if (selectedCollege !== 'All Colleges' && note.college !== selectedCollege) return false;
      if (selectedBranch !== 'All Branches' && note.branch !== selectedBranch) return false;
      if (selectedSemester !== 'All Semesters' && note.semester !== selectedSemester) return false;
      if (selectedSubject !== 'All Subjects' && note.subject !== selectedSubject) return false;
      if (selectedNoteType !== 'All Types' && note.noteType !== selectedNoteType) return false;

      if (searchQuery.trim() !== '') {
        const q = searchQuery.toLowerCase();
        const matchesTitle = note.title.toLowerCase().includes(q);
        const matchesDesc = note.description.toLowerCase().includes(q);
        const matchesCode = note.code.toLowerCase().includes(q);
        const matchesAuthor = note.author.toLowerCase().includes(q);
        const matchesSubj = note.subject.toLowerCase().includes(q);
        return matchesTitle || matchesDesc || matchesCode || matchesAuthor || matchesSubj;
      }

      return true;
    }).sort((a, b) => {
      if (selectedSortBy === 'Most Liked') return b.likesCount - a.likesCount;
      if (selectedSortBy === 'Title A-Z') return a.title.localeCompare(b.title);
      // Default: Most Recent
      return new Date(b.date).getTime() - new Date(a.date).getTime();
    });
  }, [
    notes,
    selectedCollege,
    selectedBranch,
    selectedSemester,
    selectedSubject,
    selectedNoteType,
    selectedSortBy,
    searchQuery
  ]);

  return (
    <div className="lp-container">
      {/* ── Hero Banner ────────────────────────────────────────── */}
      <section className="lp-hero">
        <h1 className="lp-hero__title">
          Turn lecture audio into
          <br />
          <span className="lp-hero__title-highlight">exam-ready notes.</span>
        </h1>

        <div className="lp-hero__cta-wrapper">
          <button className="lp-hero__cta-btn" onClick={onCreateNotebook}>
            <Plus size={20} strokeWidth={2.5} />
            <span>Create Notebook</span>
          </button>
          <span className="lp-hero__cta-subtext">Exam-ready Notes.</span>
        </div>
      </section>

      {/* ── Filter Bar ──────────────────────────────────────────── */}
      <section className="lp-filter-bar">
        <div className="lp-filter-bar__filters">
          {/* College Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">COLLEGE</span>
            <select
              className="lp-filter-select"
              value={selectedCollege}
              onChange={e => setSelectedCollege(e.target.value)}
            >
              {colleges.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Branch Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">BRANCH</span>
            <select
              className="lp-filter-select"
              value={selectedBranch}
              onChange={e => setSelectedBranch(e.target.value)}
            >
              {branches.map(b => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>

          {/* Semester Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">SEMESTER</span>
            <select
              className="lp-filter-select"
              value={selectedSemester}
              onChange={e => setSelectedSemester(e.target.value)}
            >
              {semesters.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Subject Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">SUBJECT</span>
            <select
              className="lp-filter-select"
              value={selectedSubject}
              onChange={e => setSelectedSubject(e.target.value)}
            >
              {subjects.map(sub => (
                <option key={sub} value={sub}>{sub}</option>
              ))}
            </select>
          </div>

          {/* Note Type Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">NOTE TYPE</span>
            <select
              className="lp-filter-select"
              value={selectedNoteType}
              onChange={e => setSelectedNoteType(e.target.value)}
            >
              {noteTypes.map(nt => (
                <option key={nt} value={nt}>{nt}</option>
              ))}
            </select>
          </div>

          {/* Sort By Filter */}
          <div className="lp-filter-group">
            <span className="lp-filter-group__label">SORT BY</span>
            <select
              className="lp-filter-select"
              value={selectedSortBy}
              onChange={e => setSelectedSortBy(e.target.value)}
            >
              <option value="Most Recent">Most Recent</option>
              <option value="Most Liked">Most Liked</option>
              <option value="Title A-Z">Title A-Z</option>
            </select>
          </div>
        </div>

        {/* Search Box */}
        <div className="lp-search-box">
          <Search className="lp-search-box__icon" size={16} />
          <input
            type="text"
            className="lp-search-box__input"
            placeholder="Search notes, subjects, or keywords..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>
      </section>

      {/* ── Public Notes Section ────────────────────────────────── */}
      <section>
        <div className="lp-section-header">
          <div className="lp-section-header__icon">
            <BookOpen size={20} />
          </div>
          <h2 className="lp-section-header__title">
            Public Study Notes{' '}
            <span className="lp-section-header__count">({filteredNotes.length})</span>
          </h2>
          {loading && (
            <Loader2
              size={18}
              style={{
                marginLeft: '10px',
                color: '#F59E0B',
                animation: 'spin 1s linear infinite'
              }}
            />
          )}
          <button
            onClick={fetchPublicNotes}
            title="Refresh notes from server"
            style={{
              marginLeft: 'auto',
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.82rem'
            }}
          >
            <RefreshCw size={14} />
            <span>Refresh</span>
          </button>
        </div>

        {errorMsg && (
          <div
            style={{
              padding: '10px 14px',
              marginBottom: '16px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: '8px',
              color: '#f87171',
              fontSize: '0.85rem'
            }}
          >
            {errorMsg}
          </div>
        )}

        {filteredNotes.length === 0 ? (
          <div className="lp-empty">
            <BookOpen size={36} color="#6B7280" />
            <div className="lp-empty__title">No matching public notes found</div>
            <p>Try adjusting your search criteria or resetting filters.</p>
          </div>
        ) : (
          <div className="lp-grid">
            {filteredNotes.map(note => (
              <article key={note.id} className="lp-card">
                <div className="lp-card__top">
                  <span className="lp-card__tag">{note.code}</span>
                  <span className="lp-card__date">{note.date}</span>
                </div>

                <div>
                  <h3 className="lp-card__title">{note.title}</h3>
                  <p className="lp-card__desc">{note.description}</p>
                </div>

                <div className="lp-card__footer">
                  <div className="lp-card__author">
                    <div className="lp-card__author-avatar">
                      <User size={14} />
                    </div>
                    <span>{note.author}</span>
                  </div>

                  <button
                    className="lp-card__read-btn"
                    onClick={() => onSelectNote(note)}
                  >
                    <span>Read</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default LandingPage;
