import React from 'react';
import { Home, MessageSquare, LogIn, Sparkles, User } from 'lucide-react';
import './BottomNav.css';

interface BottomNavProps {
  activeTab: 'home' | 'studio' | 'chat';
  isLoggedIn: boolean;
  onTabChange: (tab: 'home' | 'studio' | 'chat') => void;
  onOpenAuth: () => void;
  onOpenAccount?: () => void;
}

export const BottomNav: React.FC<BottomNavProps> = ({
  activeTab,
  isLoggedIn,
  onTabChange,
  onOpenAuth,
  onOpenAccount
}) => {
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

        {/* Studio Tab (If Logged In) */}
        {isLoggedIn && (
          <button
            className={`bottom-nav__item ${activeTab === 'studio' ? 'active' : ''}`}
            onClick={() => onTabChange('studio')}
          >
            <span className="bottom-nav__icon-wrap">
              <Sparkles size={18} strokeWidth={2} />
            </span>
            <span className="bottom-nav__label">Studio</span>
          </button>
        )}

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
              <User size={18} strokeWidth={2} />
            </span>
            <span className="bottom-nav__label">Account</span>
          </button>
        )}
      </div>
    </nav>
  );
};

export default BottomNav;
