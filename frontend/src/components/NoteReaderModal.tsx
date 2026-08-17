import React from 'react';
import { X, User, Calendar, BookOpen, Clock, Heart, Headphones, CheckCircle2 } from 'lucide-react';
import type { PublicNote } from '../data/mockNotes';
import './NoteReaderModal.css';

interface NoteReaderModalProps {
  note: PublicNote | null;
  onClose: () => void;
  onOpenStudio?: () => void;
}

export const NoteReaderModal: React.FC<NoteReaderModalProps> = ({
  note,
  onClose,
  onOpenStudio
}) => {
  if (!note) return null;

  return (
    <div className="nr-overlay" onClick={onClose}>
      <div className="nr-modal" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="nr-header">
          <div>
            <span className="nr-badge">{note.code} • {note.subject}</span>
            <h2 className="nr-title">{note.title}</h2>
          </div>
          <button className="nr-close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="nr-body">
          {/* Metadata Bar */}
          <div className="nr-meta">
            <div className="nr-meta-item">
              <User size={14} color="#F59E0B" />
              <span>{note.author}</span>
            </div>
            <div className="nr-meta-item">
              <Calendar size={14} />
              <span>{note.date}</span>
            </div>
            <div className="nr-meta-item">
              <Clock size={14} />
              <span>{note.readTime}</span>
            </div>
            <div className="nr-meta-item">
              <Heart size={14} color="#EF4444" fill="#EF4444" />
              <span>{note.likesCount} likes</span>
            </div>
          </div>

          {/* Audio Overview Banner */}
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              padding: '14px 18px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Headphones size={24} color="#F59E0B" />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#FFFFFF' }}>AI Lecture Audio Overview Available</div>
                <div style={{ fontSize: '0.8rem', color: '#9CA3AF' }}>Listen to 2-min audio synthesis generated from lecture recording</div>
              </div>
            </div>
            {onOpenStudio && (
              <button
                onClick={onOpenStudio}
                style={{
                  background: '#F59E0B',
                  color: '#000',
                  fontWeight: 700,
                  fontSize: '0.8rem',
                  padding: '8px 14px',
                  borderRadius: '9999px',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                Open in Studio
              </button>
            )}
          </div>

          {/* Overview */}
          <div>
            <div className="nr-section-title">
              <BookOpen size={16} />
              Overview & Breakdown
            </div>
            <div className="nr-box">
              {note.fullContent.overview}
            </div>
          </div>

          {/* Key Topics Covered */}
          <div>
            <div className="nr-section-title">
              <CheckCircle2 size={16} />
              Key Concepts & Syllabus Topics
            </div>
            <div className="nr-box">
              <ul className="nr-list">
                {note.fullContent.keyTopics.map((topic, i) => (
                  <li key={i}>{topic}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Formulas if present */}
          {note.fullContent.formulas && note.fullContent.formulas.length > 0 && (
            <div>
              <div className="nr-section-title">
                ⚡ Essential Formulas & Equations
              </div>
              <div className="nr-box" style={{ fontFamily: 'monospace', background: 'rgba(15, 18, 26, 0.9)' }}>
                <ul className="nr-list">
                  {note.fullContent.formulas.map((formula, i) => (
                    <li key={i} style={{ color: '#FCD34D' }}>{formula}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Solved PYQs */}
          <div>
            <div className="nr-section-title">
              📝 Previous Year Exam Questions (PYQs) & Worked Solutions
            </div>
            <div>
              {note.fullContent.pyqs.map((pyq, i) => (
                <div key={i} className="nr-pyq-item">
                  <span className="nr-pyq-year">{pyq.year}</span>
                  <div className="nr-pyq-q">Q: {pyq.question}</div>
                  <div className="nr-pyq-ans"><strong>Solution:</strong> {pyq.solution}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NoteReaderModal;
