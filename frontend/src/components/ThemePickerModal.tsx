import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
  Palette,
  Search,
  Check,
  X,
  Sun,
  Moon,
  Laptop,
  Sparkles,
  RotateCcw,
  CheckCircle2,
  Loader2,
  AlertCircle,
  RefreshCw
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { THEMES, type ThemeDefinition, type ThemeCategory } from '../config/themes';
import './ThemePickerModal.css';

interface ThemePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onThemeSaved?: (themeId: string) => void;
}

export const ThemePickerModal: React.FC<ThemePickerModalProps> = ({
  isOpen,
  onClose,
  onThemeSaved,
}) => {
  const {
    themeId,
    setTheme,
    syncStatus,
    syncError,
    retrySync,
  } = useTheme();

  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState<ThemeCategory>('all');
  const [savedSuccessMessage, setSavedSuccessMessage] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const initialThemeIdRef = useRef(themeId);

  // Auto-focus search input when opened and capture the initial theme before modifications
  useEffect(() => {
    if (isOpen) {
      initialThemeIdRef.current = themeId;
      setTimeout(() => {
        searchInputRef.current?.focus();
      }, 100);
    } else {
      setSearchQuery('');
      setSavedSuccessMessage(null);
    }
  }, [isOpen]);

  // Handle ESC key to close modal (keeping saved selection)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Filtered themes list
  const filteredThemes = useMemo(() => {
    return THEMES.filter(theme => {
      // Category filter
      if (activeCategory === 'dark' && theme.category !== 'dark') return false;
      if (activeCategory === 'light' && theme.category !== 'light') return false;

      // Search query filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesName = theme.name.toLowerCase().includes(q);
        const matchesDesc = theme.description.toLowerCase().includes(q);
        const matchesCategory = theme.category.toLowerCase().includes(q);
        return matchesName || matchesDesc || matchesCategory;
      }
      return true;
    });
  }, [searchQuery, activeCategory]);

  const handleSelectTheme = async (theme: ThemeDefinition) => {
    const res = await setTheme(theme.id, true);
    onThemeSaved?.(theme.id);
    if (res.success) {
      setSavedSuccessMessage(`Applied & saved: ${theme.name}`);
      setTimeout(() => {
        setSavedSuccessMessage(null);
      }, 2500);
    }
  };

  const handleRevert = async () => {
    if (initialThemeIdRef.current && initialThemeIdRef.current !== themeId) {
      await setTheme(initialThemeIdRef.current, true);
      onThemeSaved?.(initialThemeIdRef.current);
      const origName = THEMES.find(t => t.id === initialThemeIdRef.current)?.name || 'initial theme';
      setSavedSuccessMessage(`Restored ${origName}`);
      setTimeout(() => {
        setSavedSuccessMessage(null);
      }, 2500);
    }
  };

  const handleClose = () => {
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="tp-overlay" onClick={handleClose} role="dialog" aria-modal="true">
      <div className="tp-modal" onClick={e => e.stopPropagation()}>
        {/* ── Top Header ── */}
        <header className="tp-header">
          <div className="tp-header-left">
            <div className="tp-header-icon-wrap">
              <Palette size={20} className="tp-header-icon" />
            </div>
            <div>
              <div className="tp-header-title-row">
                <h2 className="tp-header-title">Color Themes</h2>
              </div>
              <p className="tp-header-subtitle">
                Select an editor-inspired palette for StudySnap AI. Click any theme to immediately apply and save it permanently.
              </p>
            </div>
          </div>

          <button
            type="button"
            className="tp-close-btn"
            onClick={handleClose}
            title="Close theme selector (Esc)"
          >
            <X size={18} />
          </button>
        </header>

        {/* ── Filter & Search Toolbar ── */}
        <div className="tp-toolbar">
          <div className="tp-search-wrap">
            <Search size={15} className="tp-search-icon" />
            <input
              ref={searchInputRef}
              type="text"
              className="tp-search-input"
              placeholder="Filter themes... (e.g. Monokai, Abyss, Solarized)"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="tp-search-clear"
                onClick={() => setSearchQuery('')}
                title="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>

          <div className="tp-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'all'}
              className={`tp-tab ${activeCategory === 'all' ? 'tp-tab-active' : ''}`}
              onClick={() => setActiveCategory('all')}
            >
              <Sparkles size={13} /> All ({THEMES.length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'dark'}
              className={`tp-tab ${activeCategory === 'dark' ? 'tp-tab-active' : ''}`}
              onClick={() => setActiveCategory('dark')}
            >
              <Moon size={13} /> Dark ({THEMES.filter(t => t.category === 'dark').length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'light'}
              className={`tp-tab ${activeCategory === 'light' ? 'tp-tab-active' : ''}`}
              onClick={() => setActiveCategory('light')}
            >
              <Sun size={13} /> Light ({THEMES.filter(t => t.category === 'light').length})
            </button>
          </div>
        </div>

        {/* ── Scrollable Theme Grid ── */}
        <div className="tp-body">
          {filteredThemes.length === 0 ? (
            <div className="tp-empty-state">
              <Palette size={32} className="tp-empty-icon" />
              <h4>No matching themes found</h4>
              <p>Try searching for a different term like "Dark+", "Modern", or "Light".</p>
              <button
                type="button"
                className="tp-empty-reset"
                onClick={() => {
                  setSearchQuery('');
                  setActiveCategory('all');
                }}
              >
                Reset Filters
              </button>
            </div>
          ) : (
            <div className="tp-grid">
              {filteredThemes.map(theme => {
                const isActiveSaved = theme.id === themeId;

                return (
                  <div
                    key={theme.id}
                    className={`tp-card ${isActiveSaved ? 'tp-card-selected tp-card-saved' : ''}`}
                    onClick={() => handleSelectTheme(theme)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleSelectTheme(theme);
                      }
                    }}
                  >
                    {/* Top Row: Title + Category Icon + Checkmark */}
                    <div className="tp-card-top">
                      <div className="tp-card-title-group">
                        <span className="tp-card-name">{theme.name}</span>
                        <span className={`tp-category-pill tp-category-${theme.category}`}>
                          {theme.category === 'system' ? (
                            <Laptop size={11} />
                          ) : theme.category === 'light' ? (
                            <Sun size={11} />
                          ) : (
                            <Moon size={11} />
                          )}
                          {theme.category.charAt(0).toUpperCase() + theme.category.slice(1)}
                        </span>
                      </div>

                      {isActiveSaved && (
                        <span className="tp-active-badge" title="Active saved theme">
                          <Check size={12} strokeWidth={3} /> Active
                        </span>
                      )}
                    </div>

                    {/* Description */}
                    <p className="tp-card-desc">{theme.description}</p>

                    {/* Swatches Row */}
                    <div className="tp-swatches-row">
                      <div className="tp-swatches-strip">
                        {theme.swatches.map((color, idx) => (
                          <span
                            key={idx}
                            className="tp-swatch-dot"
                            style={{ backgroundColor: color }}
                            title={`Color swatch ${idx + 1}: ${color}`}
                          />
                        ))}
                      </div>

                      <div className="tp-card-actions">
                        {isActiveSaved ? (
                          <span className="tp-quick-apply-btn tp-quick-apply-active" style={{ opacity: 0.9, cursor: 'default' }}>
                            Selected
                          </span>
                        ) : (
                          <button
                            type="button"
                            className="tp-quick-apply-btn"
                            onClick={e => {
                              e.stopPropagation();
                              handleSelectTheme(theme);
                            }}
                            title="Apply and set as active theme"
                          >
                            Apply
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <footer className="tp-footer">
          <div className="tp-footer-status">
            {syncStatus === 'saving' ? (
              <span className="tp-saving-text">
                <Loader2 size={14} className="animate-spin" /> Saving preference to account...
              </span>
            ) : syncStatus === 'error' && syncError ? (
              <span className="tp-error-text">
                <AlertCircle size={14} /> {syncError}
                <button
                  type="button"
                  className="tp-retry-btn"
                  onClick={async () => {
                    await retrySync();
                  }}
                  title="Retry saving theme to account"
                >
                  <RefreshCw size={11} /> Retry
                </button>
              </span>
            ) : savedSuccessMessage ? (
              <span className="tp-success-text">
                <CheckCircle2 size={14} /> {savedSuccessMessage}
              </span>
            ) : (
              <span className="tp-current-info">
                Active Theme: <strong>{THEMES.find(t => t.id === themeId)?.name}</strong>
              </span>
            )}
          </div>

          <div className="tp-footer-actions">
            {themeId !== initialThemeIdRef.current && (
              <button
                type="button"
                className="tp-cancel-btn"
                onClick={handleRevert}
                title="Revert to initial theme"
              >
                <RotateCcw size={13} /> Revert to {THEMES.find(t => t.id === initialThemeIdRef.current)?.name || 'Original'}
              </button>
            )}
            <button
              type="button"
              className="tp-save-btn"
              onClick={handleClose}
            >
              <Check size={14} /> Done
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};
