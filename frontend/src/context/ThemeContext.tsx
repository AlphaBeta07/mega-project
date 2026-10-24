import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import {
  THEMES,
  resolveTheme,
  applyThemeTokens,
  type ThemeDefinition,
} from '../config/themes';
import { updateStudentThemePreference } from '../config/supabase';

const THEME_STORAGE_KEY = 'studysnap_theme';
const THEME_PENDING_SYNC_KEY = 'studysnap_theme_pending_sync';
const DEFAULT_THEME_ID = 'dark-modern';

export type ThemeSyncStatus = 'idle' | 'saving' | 'saved' | 'error';

interface ThemeContextType {
  themeId: string;
  resolvedTheme: ThemeDefinition;
  isDark: boolean;
  previewThemeId: string | null;
  themes: ThemeDefinition[];
  syncStatus: ThemeSyncStatus;
  syncError: string | null;
  setTheme: (themeId: string, syncToRemote?: boolean) => Promise<{ success: boolean; error?: string }>;
  retrySync: () => Promise<boolean>;
  previewTheme: (themeId: string | null) => void;
  confirmPreview: () => Promise<void>;
  cancelPreview: () => void;
  setUserContext: (userId: string | null) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode; initialUserId?: string | null }> = ({
  children,
  initialUserId = null,
}) => {
  // Read cached preference or default
  const [persistedThemeId, setPersistedThemeId] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      const cached = localStorage.getItem(THEME_STORAGE_KEY);
      if (cached && THEMES.some(t => t.id === cached)) {
        return cached;
      }
    }
    return DEFAULT_THEME_ID;
  });

  const [previewThemeId, setPreviewThemeId] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<ThemeSyncStatus>('idle');
  const [syncError, setSyncError] = useState<string | null>(null);

  const [systemIsDark, setSystemIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return true;
  });

  const userIdRef = useRef<string | null>(initialUserId);
  const themeReqSequenceRef = useRef<number>(0);
  const lastThemeSelectionTimestampRef = useRef<number>(Date.now());

  // Synchronize theme to remote Supabase profile and Auth metadata
  const syncThemeToRemote = useCallback(async (targetThemeId: string, userId: string): Promise<{ success: boolean; error?: string }> => {
    setSyncStatus('saving');
    setSyncError(null);
    try {
      const res = await updateStudentThemePreference(userId, targetThemeId);
      if (res.success) {
        if (typeof window !== 'undefined') {
          localStorage.removeItem(THEME_PENDING_SYNC_KEY);
        }
        setSyncStatus('saved');
        setTimeout(() => {
          setSyncStatus(prev => (prev === 'saved' ? 'idle' : prev));
        }, 3000);
        return { success: true };
      } else {
        setSyncStatus('error');
        setSyncError(res.error || 'Failed to save preference to Supabase account');
        return { success: false, error: res.error };
      }
    } catch (err: any) {
      const errMsg = err?.message || 'Network error while persisting theme to Supabase';
      setSyncStatus('error');
      setSyncError(errMsg);
      return { success: false, error: errMsg };
    }
  }, []);

  const setUserContext = useCallback((userId: string | null) => {
    userIdRef.current = userId;
    if (userId && typeof window !== 'undefined') {
      const pending = localStorage.getItem(THEME_PENDING_SYNC_KEY);
      if (pending && THEMES.some(t => t.id === pending)) {
        syncThemeToRemote(pending, userId);
      }
    }
  }, [syncThemeToRemote]);

  // System dark-mode event listener (only affects dynamic 'system' theme)
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      setSystemIsDark(e.matches);
    };

    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  // Auto-retry pending remote sync when network connection restores
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handleOnline = () => {
      const pending = localStorage.getItem(THEME_PENDING_SYNC_KEY);
      if (pending && userIdRef.current && THEMES.some(t => t.id === pending)) {
        console.log('[ThemeContext] Online event detected. Retrying pending theme sync:', pending);
        syncThemeToRemote(pending, userIdRef.current);
      }
    };
    window.addEventListener('online', handleOnline);
    return () => window.removeEventListener('online', handleOnline);
  }, [syncThemeToRemote]);

  // Active theme calculation (preview takes precedence)
  const activeEffectiveId = previewThemeId || persistedThemeId;
  const resolvedTheme = resolveTheme(activeEffectiveId, systemIsDark);

  // Apply CSS custom properties whenever effective theme changes
  useEffect(() => {
    applyThemeTokens(resolvedTheme, true);
  }, [resolvedTheme]);

  // Set & persist theme
  const setTheme = useCallback(async (newThemeId: string, syncToRemote: boolean = true): Promise<{ success: boolean; error?: string }> => {
    if (!THEMES.some(t => t.id === newThemeId)) {
      console.warn(`[ThemeContext] Unknown theme id: ${newThemeId}`);
      return { success: false, error: `Unknown theme id: ${newThemeId}` };
    }

    const reqSeq = ++themeReqSequenceRef.current;
    lastThemeSelectionTimestampRef.current = Date.now();

    setPreviewThemeId(null);
    setPersistedThemeId(newThemeId);

    if (typeof window !== 'undefined') {
      localStorage.setItem(THEME_STORAGE_KEY, newThemeId);
      localStorage.setItem('studysnap_theme_updated_at', Date.now().toString());

      // Keep cached user object in localStorage in sync with selected theme
      try {
        const stored = localStorage.getItem('studysnap_user');
        if (stored) {
          const u = JSON.parse(stored);
          if (u && typeof u === 'object') {
            u.theme_preference = newThemeId;
            localStorage.setItem('studysnap_user', JSON.stringify(u));
          }
        }
      } catch (err) {
        console.warn('[ThemeContext] Error syncing studysnap_user cache:', err);
      }
    }

    // Synchronize to Supabase profile if user is authenticated
    if (syncToRemote) {
      if (userIdRef.current) {
        if (typeof window !== 'undefined') {
          localStorage.setItem(THEME_PENDING_SYNC_KEY, newThemeId);
        }
        const res = await syncThemeToRemote(newThemeId, userIdRef.current);
        // If a newer theme request was made while waiting, ignore this response
        if (reqSeq !== themeReqSequenceRef.current) {
          return { success: true };
        }
        return res;
      } else {
        // Signed-out user: saved locally
        setSyncStatus('idle');
        setSyncError(null);
        return { success: true };
      }
    }

    return { success: true };
  }, [syncThemeToRemote]);

  // Retry synchronization of current theme
  const retrySync = useCallback(async (): Promise<boolean> => {
    if (!userIdRef.current) {
      setSyncError('Please log in to save your theme across devices.');
      setSyncStatus('error');
      return false;
    }
    const res = await syncThemeToRemote(persistedThemeId, userIdRef.current);
    return res.success;
  }, [persistedThemeId, syncThemeToRemote]);

  // Preview interactions
  const previewTheme = useCallback((themeId: string | null) => {
    setPreviewThemeId(themeId);
  }, []);

  const confirmPreview = useCallback(async () => {
    if (previewThemeId) {
      await setTheme(previewThemeId, true);
    }
  }, [previewThemeId, setTheme]);

  const cancelPreview = useCallback(() => {
    setPreviewThemeId(null);
  }, []);

  return (
    <ThemeContext.Provider
      value={{
        themeId: persistedThemeId,
        resolvedTheme,
        isDark: resolvedTheme.isDark,
        previewThemeId,
        themes: THEMES,
        syncStatus,
        syncError,
        setTheme,
        retrySync,
        previewTheme,
        confirmPreview,
        cancelPreview,
        setUserContext,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
