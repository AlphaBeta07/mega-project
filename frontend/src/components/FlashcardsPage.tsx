import React, { useState } from 'react';
import { ArrowLeft, Loader2, BrainCircuit, RefreshCw, Shuffle, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

interface Flashcard {
  id: string;
  front: string;
  back: string;
}

interface FlashcardData {
  title: string;
  flashcards: Flashcard[];
}

interface FlashcardsPageProps {
  backendUrl: string;
  selectedSourceIds: Set<string>;
  responseLanguage: string;
  onBack: () => void;
}

export const FlashcardsPage: React.FC<FlashcardsPageProps> = ({ backendUrl, selectedSourceIds, responseLanguage, onBack }) => {
  const [data, setData] = useState<FlashcardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [knownCount, setKnownCount] = useState(0);
  const [unknownCount, setUnknownCount] = useState(0);

  const generateFlashcards = async () => {
    if (selectedSourceIds.size === 0) {
      setError("Please select at least one source document in the Studio to generate flashcards.");
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch(`${backendUrl}/api/flashcards`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: responseLanguage
        }),
      });
      const resData = await res.json();
      if (resData.success && resData.flashcards) {
        setData(resData.flashcards);
        setCurrentIndex(0);
        setIsFlipped(false);
        setKnownCount(0);
        setUnknownCount(0);
      } else {
        setError("Failed to generate flashcards.");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleFlip = () => {
    setIsFlipped(!isFlipped);
  };

  const handleNext = (known: boolean) => {
    if (known) setKnownCount(prev => prev + 1);
    else setUnknownCount(prev => prev + 1);

    // Save stats
    try {
      const statsStr = localStorage.getItem('study_analytics');
      const stats = statsStr ? JSON.parse(statsStr) : {};
      stats.flashcardsReviewed = (stats.flashcardsReviewed || 0) + 1;
      localStorage.setItem('study_analytics', JSON.stringify(stats));
    } catch (e) {}

    setIsFlipped(false);
    setTimeout(() => {
      if (data && currentIndex < data.flashcards.length - 1) {
        setCurrentIndex(prev => prev + 1);
      }
    }, 150);
  };

  const shuffleCards = () => {
    if (!data) return;
    const shuffled = [...data.flashcards].sort(() => Math.random() - 0.5);
    setData({ ...data, flashcards: shuffled });
    setCurrentIndex(0);
    setIsFlipped(false);
  };

  const restartDeck = () => {
    setCurrentIndex(0);
    setIsFlipped(false);
    setKnownCount(0);
    setUnknownCount(0);
  };

  const isCompleted = data && currentIndex >= data.flashcards.length - 1 && (knownCount + unknownCount >= data.flashcards.length);

  return (
    <div className="feature-page-container" style={{ padding: 'clamp(14px, 3.5vw, 24px)', maxWidth: '800px', margin: '0 auto', height: '100%', overflowY: 'auto' }}>
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
          marginBottom: '20px',
          transition: 'all 0.2s'
        }}
      >
        <ArrowLeft size={16} />
        Back to Studio
      </button>

      {!data && !loading && (
        <div style={{ textAlign: 'center', marginTop: 'clamp(30px, 8vh, 60px)' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(229, 176, 163, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
             <BrainCircuit size={40} style={{ color: 'var(--card-flashcards-accent, #e5b0a3)' }} />
          </div>
          <h2 style={{ fontSize: 'clamp(20px, 5vw, 24px)', fontWeight: 600, marginBottom: '12px', color: 'var(--text-primary)' }}>Generate Flashcards</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', fontSize: '14px' }}>Extract key terms and definitions for quick memorization.</p>
          <button 
            onClick={generateFlashcards}
            style={{ padding: '12px 24px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--button-primary-bg, var(--accent-color))', color: 'var(--button-primary-text, #000)', fontSize: '15px', fontWeight: 600, cursor: 'pointer' }}
          >
            Start Generating
          </button>
          {error && <p style={{ color: '#ef4444', marginTop: '16px' }}>{error}</p>}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', marginTop: 'clamp(40px, 10vh, 80px)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <Loader2 size={40} className="animate-spin" style={{ color: 'var(--accent-color)' }} />
          <h3 style={{ fontSize: 'clamp(15px, 4vw, 18px)', fontWeight: 500 }}>Extracting key terms...</h3>
        </div>
      )}

      {data && !isCompleted && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
            <h2 style={{ fontSize: 'clamp(17px, 4vw, 20px)', fontWeight: 600, margin: 0 }}>{data.title}</h2>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {currentIndex + 1} / {data.flashcards.length}
              </span>
              <button onClick={shuffleCards} className="icon-btn" title="Shuffle"><Shuffle size={18} /></button>
            </div>
          </div>

          <div 
            onClick={handleFlip}
            style={{ 
              width: '100%', 
              height: 'clamp(240px, 45vh, 340px)', 
              perspective: '1000px',
              cursor: 'pointer',
              marginBottom: '24px'
            }}
          >
            <motion.div
              initial={false}
              animate={{ rotateX: isFlipped ? 180 : 0 }}
              transition={{ duration: 0.4, type: 'spring', stiffness: 200, damping: 20 }}
              style={{
                width: '100%',
                height: '100%',
                position: 'relative',
                transformStyle: 'preserve-3d'
              }}
            >
              {/* Front */}
              <div style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                backfaceVisibility: 'hidden',
                backgroundColor: 'var(--bg-panel)',
                borderRadius: '16px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 'clamp(18px, 4vw, 32px)',
                textAlign: 'center',
                overflowY: 'auto'
              }}>
                <span style={{ position: 'absolute', top: '14px', left: '16px', fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>Front</span>
                <div style={{ fontSize: 'clamp(18px, 4.5vw, 24px)', fontWeight: 600, color: 'var(--text-primary)', wordBreak: 'break-word' }}>
                  <ReactMarkdown>{data.flashcards[currentIndex].front}</ReactMarkdown>
                </div>
              </div>

              {/* Back */}
              <div style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                backfaceVisibility: 'hidden',
                backgroundColor: 'var(--surface-elevated, var(--bg-button))',
                borderRadius: '16px',
                border: '1px solid var(--accent-primary, var(--accent-color))',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 'clamp(18px, 4vw, 32px)',
                textAlign: 'center',
                transform: 'rotateX(180deg)',
                overflowY: 'auto'
              }}>
                <span style={{ position: 'absolute', top: '14px', left: '16px', fontSize: '11px', color: 'var(--accent-primary, var(--accent-color))', textTransform: 'uppercase', letterSpacing: '1px' }}>Back</span>
                <div style={{ fontSize: 'clamp(15px, 3.8vw, 19px)', lineHeight: '1.6', color: 'var(--text-primary)', wordBreak: 'break-word' }}>
                  <ReactMarkdown>{data.flashcards[currentIndex].back}</ReactMarkdown>
                </div>
              </div>
            </motion.div>
          </div>

          <div style={{ display: 'flex', gap: '12px', width: '100%', flexWrap: 'wrap' }}>
            <button 
              onClick={() => handleNext(false)}
              style={{ flex: 1, minWidth: '130px', padding: 'clamp(12px, 3vw, 16px)', borderRadius: '12px', border: '1px solid #ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', fontSize: 'clamp(14px, 3.5vw, 16px)', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              <XCircle size={18} /> Still Learning
            </button>
            <button 
              onClick={() => handleNext(true)}
              style={{ flex: 1, minWidth: '130px', padding: 'clamp(12px, 3vw, 16px)', borderRadius: '12px', border: '1px solid #10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#10b981', fontSize: 'clamp(14px, 3.5vw, 16px)', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              <CheckCircle size={18} /> Got It
            </button>
          </div>
        </div>
      )}

      {isCompleted && data && (
        <div style={{ textAlign: 'center', backgroundColor: 'var(--bg-panel)', borderRadius: '16px', border: '1px solid var(--border-color)', padding: '48px 32px' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(229, 176, 163, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
             <BrainCircuit size={40} style={{ color: '#e5b0a3' }} />
          </div>
          <h2 style={{ fontSize: '28px', fontWeight: 600, marginBottom: '8px' }}>Deck Finished!</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '16px', marginBottom: '32px' }}>You knew {knownCount} out of {data.flashcards.length} cards.</p>
          
          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
            <button 
              onClick={restartDeck}
              style={{ padding: '12px 24px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '15px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              <RefreshCw size={16} /> Review Again
            </button>
            <button 
              onClick={generateFlashcards}
              style={{ padding: '12px 24px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--bg-button)', color: 'var(--text-primary)', fontSize: '15px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              Generate New Deck
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
