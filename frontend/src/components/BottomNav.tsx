import React from 'react';
import { Home, MessageSquare, LogIn, User } from 'lucide-react';
import './BottomNav.css';

interface BottomNavProps {
  activeTab: 'home' | 'chat' | 'infographic' | 'quiz' | 'audio' | 'mindmap' | 'reports' | 'flashcards' | '';
  isLoggedIn: boolean;
  onTabChange: (tab: any) => void;
  onOpenAuth: () => void;
  onOpenAccount?: () => void;
  currentUser?: any;
}

export const BottomNav: React.FC<BottomNavProps> = ({
  activeTab,
  isLoggedIn,
  onTabChange,
  onOpenAuth,
  onOpenAccount,
  currentUser
}) => {
  const avatarUrl = currentUser?.avatar_url;
  const initial = (currentUser?.full_name?.charAt(0) || currentUser?.name?.charAt(0) || 'S').toUpperCase();

  return (
    <nav className="bottom-nav" aria-label="Main navigation">
      <div className="bottom-nav__inner">
        {/* Home Tab */}
        <button
          className={`bottom-nav__item ${activeTab === 'home' ? 'active' : ''}`}
          onClick={() => onTabChange('home')}
        >
          <span className="bottom-nav__icon-wrap">
            <Home size={18} strokeWidth={2} />
          </span>
          <span className="bottom-nav__label">Home</span>
        </button>

        {/* Chat Tab */}
        <button
          className={`bottom-nav__item ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => {
            if (!isLoggedIn) {
              onOpenAuth();
            } else {
              onTabChange('chat');
            }
          }}
        >
          <span className="bottom-nav__icon-wrap">
            <MessageSquare size={18} strokeWidth={2} />
          </span>
          <span className="bottom-nav__label">Chat</span>
        </button>

        {/* Log In / Profile Button */}
        {!isLoggedIn ? (
          <button
            className="bottom-nav__item"
            onClick={onOpenAuth}
          >
            <span className="bottom-nav__icon-wrap">
              <LogIn size={18} strokeWidth={2} />
            </span>
            <span className="bottom-nav__label">Log In</span>
          </button>
        ) : (
          <button
            className="bottom-nav__item"
            onClick={() => {
              if (onOpenAccount) onOpenAccount();
              else onOpenAuth();
            }}
            title="Account & Profile Options"
          >
            <span className="bottom-nav__icon-wrap">
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt={currentUser?.full_name || 'Account'}
                  style={{
                    width: '22px',
                    height: '22px',
                    borderRadius: '50%',
                    objectFit: 'cover',
                    border: '1.5px solid #F59E0B'
                  }}
                />
              ) : currentUser ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '22px',
                    height: '22px',
                    borderRadius: '50%',
                    backgroundColor: '#D97706',
                    color: '#FFFFFF',
                    fontSize: '11px',
                    fontWeight: 700
                  }}
                >
                  {initial}
                </span>
              ) : (
                <User size={18} strokeWidth={2} />
              )}
            </span>
            {/* <span className="bottom-nav__label">Account</span> */}
          </button>
        )}
      </div>
    </nav>
  );
};

export default BottomNav;
