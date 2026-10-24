import React, { useEffect, useState } from 'react';
import { ArrowLeft, TrendingUp, Target, BrainCircuit, Activity, Award } from 'lucide-react';

interface StudyAnalytics {
  quizzesAttempted: number;
  totalQuestionsAnswered: number;
  totalScore: number;
  flashcardsReviewed: number;
  sessions: number;
}

interface ReportsPageProps {
  onBack: () => void;
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ onBack }) => {
  const [stats, setStats] = useState<StudyAnalytics>({
    quizzesAttempted: 0,
    totalQuestionsAnswered: 0,
    totalScore: 0,
    flashcardsReviewed: 0,
    sessions: 0
  });

  useEffect(() => {
    // Load from local storage
    try {
      const statsStr = localStorage.getItem('study_analytics');
      if (statsStr) {
        setStats(JSON.parse(statsStr));
      }
    } catch (e) {
      console.error("Failed to load stats", e);
    }
    
    // Increment sessions
    try {
      const statsStr = localStorage.getItem('study_analytics');
      const currentStats = statsStr ? JSON.parse(statsStr) : { quizzesAttempted: 0, totalQuestionsAnswered: 0, totalScore: 0, flashcardsReviewed: 0, sessions: 0 };
      currentStats.sessions = (currentStats.sessions || 0) + 1;
      localStorage.setItem('study_analytics', JSON.stringify(currentStats));
      setStats(currentStats);
    } catch(e) {}
  }, []);

  const averageScore = stats.totalQuestionsAnswered > 0 
    ? Math.round((stats.totalScore / stats.totalQuestionsAnswered) * 100) 
    : 0;

  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto', height: '100%', overflowY: 'auto' }}>
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
          marginBottom: '32px',
          transition: 'all 0.2s'
        }}
      >
        <ArrowLeft size={16} />
        Back to Studio
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
        <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: 'rgba(223, 201, 138, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <TrendingUp size={28} style={{ color: '#dfc98a' }} />
        </div>
        <div>
          <h1 style={{ fontSize: '28px', fontWeight: 600, margin: '0 0 4px 0', color: 'var(--text-primary)' }}>Study Analytics</h1>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '15px' }}>Track your learning progress and activity.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        
        {/* Stat Card 1 */}
        <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(163, 216, 211, 0.1)' }}>
              <Target size={20} style={{ color: '#a3d8d3' }} />
            </div>
            <span style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-secondary)' }}>Quiz Performance</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '36px', fontWeight: 700, color: 'var(--text-primary)' }}>{averageScore}%</span>
            <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>average score</span>
          </div>
          <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-button)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ width: `${averageScore}%`, height: '100%', backgroundColor: '#a3d8d3', borderRadius: '3px' }} />
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Quizzes Attempted: <strong>{stats.quizzesAttempted}</strong></span>
            <span>Questions Answered: <strong>{stats.totalQuestionsAnswered}</strong></span>
          </div>
        </div>

        {/* Stat Card 2 */}
        <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(229, 176, 163, 0.1)' }}>
              <BrainCircuit size={20} style={{ color: '#e5b0a3' }} />
            </div>
            <span style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-secondary)' }}>Memory Retention</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '36px', fontWeight: 700, color: 'var(--text-primary)' }}>{stats.flashcardsReviewed}</span>
            <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>flashcards reviewed</span>
          </div>
          <p style={{ margin: 'auto 0 0 0', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Regularly reviewing flashcards improves long-term retention. Try generating a new deck to test your knowledge!
          </p>
        </div>

        {/* Stat Card 3 */}
        <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(168, 199, 250, 0.1)' }}>
              <Activity size={20} style={{ color: '#a8c7fa' }} />
            </div>
            <span style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-secondary)' }}>Study Sessions</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '36px', fontWeight: 700, color: 'var(--text-primary)' }}>{stats.sessions}</span>
            <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>sessions</span>
          </div>
           <p style={{ margin: 'auto 0 0 0', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Consistent study sessions are key to success. Keep up the good work!
          </p>
        </div>
      </div>

      <div style={{ backgroundColor: 'var(--bg-panel)', padding: '32px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '24px' }}>
        <div style={{ padding: '20px', borderRadius: '50%', backgroundColor: 'rgba(251, 191, 36, 0.1)' }}>
          <Award size={48} style={{ color: '#FBBF24' }} />
        </div>
        <div>
          <h3 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 8px 0' }}>Keep learning!</h3>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.5 }}>
            The more you interact with the AI tools, take quizzes, and review flashcards, the more your analytics will grow. Try adding more sources to unlock new insights.
          </p>
        </div>
      </div>

    </div>
  );
};
