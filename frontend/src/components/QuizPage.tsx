import React, { useState } from 'react';
import { ArrowLeft, CheckCircle, XCircle, RefreshCw, Loader2, PlayCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

interface QuizData {
  title: string;
  questions: QuizQuestion[];
}

interface QuizPageProps {
  backendUrl: string;
  selectedSourceIds: Set<string>;
  responseLanguage: string;
  onBack: () => void;
}

export const QuizPage: React.FC<QuizPageProps> = ({ backendUrl, selectedSourceIds, responseLanguage, onBack }) => {
  const [quizData, setQuizData] = useState<QuizData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [isAnswered, setIsAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [showResults, setShowResults] = useState(false);

  const generateQuiz = async () => {
    if (selectedSourceIds.size === 0) {
      setError("Please select at least one source document in the Studio to generate a quiz.");
      return;
    }
    setLoading(true);
    setError(null);
    setQuizData(null);
    try {
      const res = await fetch(`${backendUrl}/api/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: responseLanguage
        }),
      });
      const data = await res.json();
      if (data.success && data.quiz) {
        setQuizData(data.quiz);
        setCurrentQuestionIndex(0);
        setSelectedOption(null);
        setIsAnswered(false);
        setScore(0);
        setShowResults(false);
      } else {
        setError("Failed to generate quiz data.");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleOptionClick = (index: number) => {
    if (isAnswered) return;
    setSelectedOption(index);
    setIsAnswered(true);
    
    if (quizData && index === quizData.questions[currentQuestionIndex].correctIndex) {
      setScore(prev => prev + 1);
    }
  };

  const handleNext = () => {
    if (!quizData) return;
    if (currentQuestionIndex < quizData.questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setSelectedOption(null);
      setIsAnswered(false);
    } else {
      setShowResults(true);
      
      // Save stats to localStorage for Reports
      try {
        const statsStr = localStorage.getItem('study_analytics');
        const stats = statsStr ? JSON.parse(statsStr) : { quizzesAttempted: 0, totalQuestionsAnswered: 0, totalScore: 0 };
        stats.quizzesAttempted = (stats.quizzesAttempted || 0) + 1;
        stats.totalQuestionsAnswered = (stats.totalQuestionsAnswered || 0) + quizData.questions.length;
        stats.totalScore = (stats.totalScore || 0) + score + (selectedOption === quizData.questions[currentQuestionIndex].correctIndex ? 1 : 0);
        localStorage.setItem('study_analytics', JSON.stringify(stats));
      } catch (e) {
        console.error("Failed to save stats", e);
      }
    }
  };

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

      {!quizData && !loading && (
        <div style={{ textAlign: 'center', marginTop: 'clamp(30px, 8vh, 60px)' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(163, 216, 211, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
             <PlayCircle size={40} style={{ color: '#a3d8d3' }} />
          </div>
          <h2 style={{ fontSize: 'clamp(20px, 5vw, 24px)', fontWeight: 600, marginBottom: '12px' }}>Generate a Quiz</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', fontSize: '14px' }}>Test your knowledge based on the selected study materials.</p>
          <button 
            onClick={generateQuiz}
            style={{ padding: '12px 24px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '15px', fontWeight: 600, cursor: 'pointer' }}
          >
            Start Generating
          </button>
          {error && <p style={{ color: '#ef4444', marginTop: '16px' }}>{error}</p>}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', marginTop: 'clamp(40px, 10vh, 80px)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <Loader2 size={40} className="animate-spin" style={{ color: 'var(--accent-color)' }} />
          <h3 style={{ fontSize: 'clamp(15px, 4vw, 18px)', fontWeight: 500 }}>Analyzing sources and generating questions...</h3>
        </div>
      )}

      {quizData && !showResults && !loading && (
        <div style={{ backgroundColor: 'var(--bg-panel)', borderRadius: '16px', border: '1px solid var(--border-color)', padding: 'clamp(16px, 4vw, 32px)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap', marginBottom: '24px' }}>
            <h2 style={{ fontSize: 'clamp(17px, 4vw, 20px)', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>{quizData.title}</h2>
            <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>Question {currentQuestionIndex + 1} of {quizData.questions.length}</span>
          </div>
          
          <div style={{ fontSize: 'clamp(15px, 3.8vw, 18px)', lineHeight: '1.6', marginBottom: '28px', wordBreak: 'break-word' }}>
            <ReactMarkdown>{quizData.questions[currentQuestionIndex].question}</ReactMarkdown>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {quizData.questions[currentQuestionIndex].options.map((opt, idx) => {
              const isCorrect = idx === quizData.questions[currentQuestionIndex].correctIndex;
              const isSelected = selectedOption === idx;
              
              let bgColor = 'var(--bg-button)';
              let borderColor = 'var(--border-color)';
              if (isAnswered) {
                if (isCorrect) {
                  bgColor = 'rgba(16, 185, 129, 0.1)';
                  borderColor = '#10b981';
                } else if (isSelected) {
                  bgColor = 'rgba(239, 68, 68, 0.1)';
                  borderColor = '#ef4444';
                }
              } else if (isSelected) {
                bgColor = 'rgba(168, 199, 250, 0.1)';
                borderColor = 'var(--accent-color)';
              }

              return (
                <button
                  key={idx}
                  onClick={() => handleOptionClick(idx)}
                  disabled={isAnswered}
                  style={{
                    padding: 'clamp(12px, 3vw, 16px) clamp(14px, 3.5vw, 20px)',
                    borderRadius: '12px',
                    border: `1px solid ${borderColor}`,
                    backgroundColor: bgColor,
                    color: 'var(--text-primary)',
                    fontSize: 'clamp(14px, 3.5vw, 16px)',
                    textAlign: 'left',
                    cursor: isAnswered ? 'default' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    transition: 'all 0.2s',
                    wordBreak: 'break-word'
                  }}
                >
                  <span style={{ flex: 1 }}>{opt}</span>
                  {isAnswered && isCorrect && <CheckCircle size={20} color="#10b981" style={{ flexShrink: 0 }} />}
                  {isAnswered && isSelected && !isCorrect && <XCircle size={20} color="#ef4444" style={{ flexShrink: 0 }} />}
                </button>
              );
            })}
          </div>

          {isAnswered && (
            <div style={{ marginTop: '20px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '15px', color: 'var(--text-primary)' }}>Explanation</h4>
              <p style={{ margin: 0, fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {quizData.questions[currentQuestionIndex].explanation}
              </p>
            </div>
          )}

          {isAnswered && (
            <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end' }}>
              <button 
                onClick={handleNext}
                style={{ width: '100%', maxWidth: '240px', padding: '12px 28px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', fontSize: '15px', fontWeight: 600, cursor: 'pointer' }}
              >
                {currentQuestionIndex < quizData.questions.length - 1 ? 'Next Question' : 'View Results'}
              </button>
            </div>
          )}
        </div>
      )}

      {showResults && quizData && (
        <div style={{ textAlign: 'center', backgroundColor: 'var(--bg-panel)', borderRadius: '16px', border: '1px solid var(--border-color)', padding: '48px 32px' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
             <CheckCircle size={40} style={{ color: '#10b981' }} />
          </div>
          <h2 style={{ fontSize: '28px', fontWeight: 600, marginBottom: '8px' }}>Quiz Complete!</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '16px', marginBottom: '32px' }}>You scored {score} out of {quizData.questions.length}</p>
          
          <button 
            onClick={generateQuiz}
            style={{ padding: '12px 24px', borderRadius: '8px', backgroundColor: 'transparent', color: 'var(--text-primary)', fontSize: '15px', fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '8px', border: '1px solid var(--border-color)' }}
          >
            <RefreshCw size={16} /> New Quiz
          </button>
        </div>
      )}
    </div>
  );
};
