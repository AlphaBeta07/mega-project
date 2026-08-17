import React, { useState } from 'react';
import { X, Sparkles } from 'lucide-react';
import './AuthModal.css';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: { name: string; email: string; college: string }) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  onLoginSuccess
}) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLoginSuccess({
      name: email.split('@')[0] || 'Student User',
      email: email || 'student@iitb.ac.in',
      college: 'IIT Bombay'
    });
  };

  const handleQuickDemoLogin = () => {
    onLoginSuccess({
      name: 'Rohan Sharma',
      email: 'rohan.sharma@iitb.ac.in',
      college: 'IIT Bombay'
    });
  };

  return (
    <div className="auth-overlay" onClick={onClose}>
      <div className="auth-modal" onClick={e => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose}>
          <X size={20} />
        </button>

        <h2 className="auth-title">Log in to StudySnap</h2>
        <p className="auth-subtitle">Turn your lecture audio into exam-ready notes</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div>
            <label className="auth-label">College Email</label>
            <input
              type="email"
              className="auth-input"
              placeholder="name@college.edu.in"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="auth-label">Password</label>
            <input
              type="password"
              className="auth-input"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="auth-submit-btn">
            Log In
          </button>
        </form>

        <div style={{ margin: '16px 0 8px 0', textAlign: 'center', fontSize: '0.8rem', color: '#6B7280' }}>
          ────── OR ──────
        </div>

        <button className="auth-demo-btn" onClick={handleQuickDemoLogin}>
          <Sparkles size={16} />
          Quick Demo Login (Instant Access)
        </button>
      </div>
    </div>
  );
};

export default AuthModal;
