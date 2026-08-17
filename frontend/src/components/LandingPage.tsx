import React, { useState, useMemo } from 'react';
import { Search, BookOpen, Plus, User, ArrowRight } from 'lucide-react';
import { MOCK_PUBLIC_NOTES } from '../data/mockNotes';
import type { PublicNote } from '../data/mockNotes';
import './LandingPage.css';

interface LandingPageProps {
  onCreateNotebook: () => void;
  onSelectNote: (note: PublicNote) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onCreateNotebook,
  onSelectNote,
}) => {
  // Filter States
  const [selectedCollege, setSelectedCollege] = useState('All Colleges');
  const [selectedBranch, setSelectedBranch] = useState('All Branches');
  const [selectedSemester, setSelectedSemester] = useState('All Semesters');
  const [selectedSubject, setSelectedSubject] = useState('All Subjects');
  const [selectedNoteType, setSelectedNoteType] = useState('All Types');
  const [selectedSortBy, setSelectedSortBy] = useState('Most Recent');
  const [searchQuery, setSearchQuery] = useState('');

  // Extract unique filter dropdown options dynamically from mock notes
  const colleges = useMemo(() => ['All Colleges', ...Array.from(new Set(MOCK_PUBLIC_NOTES.map(n => n.college)))], []);
  const branches = useMemo(() => ['All Branches', ...Array.from(new Set(MOCK_PUBLIC_NOTES.map(n => n.branch)))], []);
  const semesters = useMemo(() => ['All Semesters', ...Array.from(new Set(MOCK_PUBLIC_NOTES.map(n => n.semester)))], []);
  const subjects = useMemo(() => ['All Subjects', ...Array.from(new Set(MOCK_PUBLIC_NOTES.map(n => n.subject)))], []);
  const noteTypes = useMemo(() => ['All Types', 'Lecture Notes', 'PYQ & Solutions', 'Formula Sheet', 'Lab Manual'], []);

  // Filter & Sort Logic
  const filteredNotes = useMemo(() => {
    return MOCK_PUBLIC_NOTES.filter(note => {
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
        </div>

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
