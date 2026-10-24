import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import {
  THEMES,
  resolveTheme,
  applyThemeTokens,
  type ThemeDefinition,
} from '../config/themes';
import { updateStudentProfile } from '../config/supabase';

const THEME_STORAGE_KEY = 'studysnap_theme';
const DEFAULT_THEME_ID = 'dark-modern';

interface ThemeContextType {
  themeId: string;
  resolvedTheme: ThemeDefinition;
  isDark: boolean;
  previewThemeId: string | null;
  themes: ThemeDefinition[];
  setTheme: (themeId: string, syncToRemote?: boolean) => Promise<void>;
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
  const [systemIsDark, setSystemIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return true;
  });

  const userIdRef = useRef<string | null>(initialUserId);

  const setUserContext = useCallback((userId: string | null) => {
    userIdRef.current = userId;
  }, []);

  // System dark-mode event listener
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      setSystemIsDark(e.matches);
    };

    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  // Active theme calculation (preview takes precedence)
  const activeEffectiveId = previewThemeId || persistedThemeId;
  const resolvedTheme = resolveTheme(activeEffectiveId, systemIsDark);

  // Apply CSS custom properties whenever effective theme changes
  useEffect(() => {
    applyThemeTokens(resolvedTheme, true);
  }, [resolvedTheme]);

  // Set & persist theme
  const setTheme = useCallback(async (newThemeId: string, syncToRemote: boolean = true) => {
    if (!THEMES.some(t => t.id === newThemeId)) {
      console.warn(`[ThemeContext] Unknown theme id: ${newThemeId}`);
      return;
    }

    setPreviewThemeId(null);
    setPersistedThemeId(newThemeId);
    if (typeof window !== 'undefined') {
      localStorage.setItem(THEME_STORAGE_KEY, newThemeId);
    }

    // Synchronize to Supabase profile if user is authenticated
    if (syncToRemote && userIdRef.current) {
      try {
        await updateStudentProfile(userIdRef.current, { theme_preference: newThemeId });
      } catch (err) {
        console.warn('[ThemeContext] Could not sync theme preference to remote profile:', err);
      }
    }
  }, []);

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
        setTheme,
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
