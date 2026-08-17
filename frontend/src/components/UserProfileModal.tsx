import React from 'react';
import { X, LogOut, GraduationCap, Mail, ShieldCheck } from 'lucide-react';
import './UserProfileModal.css';

interface UserProfileModalProps {
  isOpen: boolean;
  user: { name: string; email: string; college: string } | null;
  onClose: () => void;
  onLogout: () => void;
}

export const UserProfileModal: React.FC<UserProfileModalProps> = ({
  isOpen,
  user,
  onClose,
  onLogout
}) => {
  if (!isOpen) return null;

  const name = user?.name || 'Rohan Sharma';
  const email = user?.email || 'rohan.sharma@iitb.ac.in';
  const college = user?.college || 'IIT Bombay';

  return (
    <div className="up-overlay" onClick={onClose}>
      <div className="up-modal" onClick={e => e.stopPropagation()}>
        <button className="up-close" onClick={onClose}>
          <X size={18} />
        </button>

        <div className="up-header">
          <div className="up-avatar">
            {name.charAt(0)}
          </div>
          <h2 className="up-name">{name}</h2>
          <span className="up-email">{email}</span>
          <span className="up-college-badge">{college}</span>
        </div>

        <div className="up-info-list">
          <div className="up-info-item">
            <span className="up-info-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <GraduationCap size={16} /> Branch & Sem
            </span>
            <span className="up-info-val">CS • Semester 5</span>
          </div>

          <div className="up-info-item">
            <span className="up-info-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Mail size={16} /> Account Status
            </span>
            <span className="up-info-val" style={{ color: '#4ADE80' }}>Verified Student</span>
          </div>

          <div className="up-info-item">
            <span className="up-info-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ShieldCheck size={16} /> Plan Type
            </span>
            <span className="up-info-val" style={{ color: '#FBBF24' }}>Student Pro Edition</span>
          </div>
        </div>

        <button
          className="up-logout-btn"
          onClick={() => {
            onLogout();
            onClose();
          }}
        >
          <LogOut size={16} />
          Log Out
        </button>
      </div>
    </div>
  );
};

export default UserProfileModal;
