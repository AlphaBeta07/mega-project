/**
 * StudySnap AI — VS Code–Inspired Theme System
 * Curated collection of editor-inspired color themes.
 */

export type ThemeCategory = 'all' | 'dark' | 'light';

export interface ThemeTokenMap {
  '--color-bg-base': string;
  '--color-bg-surface': string;
  '--color-bg-elevated': string;
  '--color-bg-overlay': string;
  '--color-bg-sunken': string;
  '--color-bg-input': string;
  '--color-bg-surface-glass': string;

  '--color-text-primary': string;
  '--color-text-secondary': string;
  '--color-text-muted': string;
  '--color-text-disabled': string;
  '--color-text-inverse': string;
  '--color-text-accent': string;

  '--color-border-subtle': string;
  '--color-border-default': string;
  '--color-border-strong': string;
  '--color-border-focus': string;
  '--color-border-accent': string;

  '--color-accent-500': string;
  '--color-accent-400': string;
  '--color-cta-bg': string;
  '--color-cta-bg-hover': string;
  '--color-cta-bg-active': string;
  '--color-cta-text': string;

  '--color-nav-active-bg': string;
  '--color-nav-active-text': string;

  '--color-scrollbar-thumb': string;
  '--color-scrollbar-track': string;

  // Legacy aliases
  '--bg-main': string;
  '--bg-panel': string;
  '--bg-panel-hover': string;
  '--bg-button': string;
  '--bg-button-hover': string;
  '--bg-input': string;
  '--text-primary': string;
  '--text-secondary': string;
  '--text-tertiary': string;
  '--accent-color': string;
  '--border-color': string;
  '--border-light': string;

  // Profile modal aliases
  '--up-bg-modal': string;
  '--up-bg-header': string;
  '--up-bg-surface': string;
  '--up-bg-surface-hover': string;
  '--up-border': string;
  '--up-border-subtle': string;
  '--up-text-primary': string;
  '--up-text-secondary': string;
  '--up-text-muted': string;
  '--up-accent': string;
  '--up-accent-hover': string;
  '--up-accent-subtle': string;
  // Standardized workspace tokens
  '--app-bg'?: string;
  '--surface-primary'?: string;
  '--surface-secondary'?: string;
  '--surface-elevated'?: string;
  '--surface-input'?: string;
  '--text-muted'?: string;
  '--border-default'?: string;
  '--border-subtle'?: string;
  '--accent-primary'?: string;
  '--accent-hover'?: string;
  '--accent-contrast'?: string;
  '--button-primary-bg'?: string;
  '--button-primary-text'?: string;
  '--button-secondary-bg'?: string;
  '--hover-bg'?: string;
  '--selected-bg'?: string;
  '--focus-ring'?: string;
  '--success'?: string;
  '--warning'?: string;
  '--danger'?: string;
  '--chat-user-bg'?: string;
  '--chat-assistant-bg'?: string;
  '--chat-composer-bg'?: string;
  '--source-card-bg'?: string;
  '--sidebar-bg'?: string;

  // Particle background tokens
  '--particle-color-1'?: string;
  '--particle-color-2'?: string;
  '--particle-color-3'?: string;
  '--particle-color-4'?: string;
  '--particle-opacity'?: string;
  '--particle-glow'?: string;

  // Extensibility index signature
  [key: string]: string | undefined;
}

export interface ParticleColorStop {
  rgb: [number, number, number];
  weight: number;
}

export interface ThemeDefinition {
  id: string;
  name: string;
  category: 'dark' | 'light' | 'system';
  isDark: boolean;
  description: string;
  authorNote?: string;
  swatches: string[]; // 5 preview colors: [bg, surface, accent, secondaryAccent, text]
  tokens: ThemeTokenMap;
  particlePalette: ParticleColorStop[];
}

export const THEMES: ThemeDefinition[] = [
  // ── 0. System Default ──
  {
    id: 'system',
    name: 'System Default',
    category: 'system',
    isDark: true, // evaluated dynamically at runtime
    description: 'Automatically match your operating system theme preference (Light or Dark).',
    swatches: ['#0B0E17', '#111622', '#F4A900', '#3B82F6', '#F0F3F8'],
    particlePalette: [
      { rgb: [59, 130, 246], weight: 0.35 },
      { rgb: [99, 102, 241], weight: 0.25 },
      { rgb: [139, 92, 246], weight: 0.20 },
      { rgb: [168, 85, 247], weight: 0.12 },
      { rgb: [236, 72, 153], weight: 0.08 },
    ],
    tokens: {} as ThemeTokenMap, // dynamically populated from dark-modern or light-modern
  },

  // ── 1. Dark Modern (StudySnap AI Oxford Navy & VS Code Modern Dark) ──
  {
    id: 'dark-modern',
    name: 'Dark Modern',
    category: 'dark',
    isDark: true,
    description: 'The polished modern dark theme with deep Oxford Blue hues and vibrant amber accents.',
    swatches: ['#0B0E17', '#111622', '#F4A900', '#3B7BBB', '#F0F3F8'],
    particlePalette: [
      { rgb: [59, 123, 187], weight: 0.35 },
      { rgb: [99, 102, 241], weight: 0.25 },
      { rgb: [139, 92, 246], weight: 0.20 },
      { rgb: [244, 169, 0], weight: 0.12 },
      { rgb: [236, 72, 153], weight: 0.08 },
    ],
    tokens: {
      '--color-bg-base': '#0B0E17',
      '--color-bg-surface': '#111622',
      '--color-bg-elevated': '#191E2E',
      '--color-bg-overlay': '#232A3E',
      '--color-bg-sunken': '#07090F',
      '--color-bg-input': '#0F1420',
      '--color-bg-surface-glass': 'rgba(17, 22, 34, 0.88)',

      '--color-text-primary': '#F0F3F8',
      '--color-text-secondary': '#B8C2D6',
      '--color-text-muted': '#687288',
      '--color-text-disabled': '#313A52',
      '--color-text-inverse': '#0B0E17',
      '--color-text-accent': '#F7BC2E',

      '--color-border-subtle': '#191E2E',
      '--color-border-default': '#232A3E',
      '--color-border-strong': '#313A52',
      '--color-border-focus': '#F4A900',
      '--color-border-accent': '#B36A00',

      '--color-accent-500': '#F4A900',
      '--color-accent-400': '#F7BC2E',
      '--color-cta-bg': '#F4A900',
      '--color-cta-bg-hover': '#F7BC2E',
      '--color-cta-bg-active': '#FACC60',
      '--color-cta-text': '#0B0E17',

      '--color-nav-active-bg': 'rgba(244, 169, 0, 0.12)',
      '--color-nav-active-text': '#F7BC2E',

      '--color-scrollbar-thumb': '#2A334B',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#0B0E17',
      '--bg-panel': '#111622',
      '--bg-panel-hover': '#191E2E',
      '--bg-button': '#191E2E',
      '--bg-button-hover': '#232A3E',
      '--bg-input': '#0F1420',
      '--text-primary': '#F0F3F8',
      '--text-secondary': '#B8C2D6',
      '--text-tertiary': '#687288',
      '--accent-color': '#F7BC2E',
      '--border-color': '#232A3E',
      '--border-light': 'rgba(255, 255, 255, 0.08)',

      '--up-bg-modal': '#111622',
      '--up-bg-header': '#0B0E17',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.03)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.06)',
      '--up-border': '#232A3E',
      '--up-border-subtle': '#191E2E',
      '--up-text-primary': '#F0F3F8',
      '--up-text-secondary': '#B8C2D6',
      '--up-text-muted': '#687288',
      '--up-accent': '#F4A900',
      '--up-accent-hover': '#F7BC2E',
      '--up-accent-subtle': 'rgba(244, 169, 0, 0.10)',
      '--up-accent-border': 'rgba(244, 169, 0, 0.25)',
    },
  },

  // ── 2. Dark+ (Classic Visual Studio Code Dark+) ──
  {
    id: 'dark-plus',
    name: 'Dark+',
    category: 'dark',
    isDark: true,
    description: 'The iconic default VS Code Dark+ theme with neutral charcoal surfaces and classic blue accents.',
    swatches: ['#1E1E1E', '#252526', '#0E639C', '#4EC9B0', '#CCCCCC'],
    particlePalette: [
      { rgb: [59, 130, 246], weight: 0.35 },  // Electric Blue
      { rgb: [99, 102, 241], weight: 0.25 },  // Indigo
      { rgb: [139, 92, 246], weight: 0.20 },  // Violet
      { rgb: [168, 85, 247], weight: 0.12 },  // Subtle Purple
      { rgb: [236, 72, 153], weight: 0.08 },  // Occasional Pink
    ],
    tokens: {
      '--color-bg-base': '#181818',
      '--color-bg-surface': '#1F1F1F',
      '--color-bg-elevated': '#282828',
      '--color-bg-overlay': '#333333',
      '--color-bg-sunken': '#121212',
      '--color-bg-input': '#222222',
      '--color-bg-surface-glass': 'rgba(31, 31, 31, 0.90)',

      '--color-text-primary': '#CCCCCC',
      '--color-text-secondary': '#9D9D9D',
      '--color-text-muted': '#6E6E6E',
      '--color-text-disabled': '#454545',
      '--color-text-inverse': '#181818',
      '--color-text-accent': '#3794FF',

      '--color-border-subtle': '#2B2B2B',
      '--color-border-default': '#383838',
      '--color-border-strong': '#4A4A4A',
      '--color-border-focus': '#0E639C',
      '--color-border-accent': '#1177BB',

      '--color-accent-500': '#0E639C',
      '--color-accent-400': '#3794FF',
      '--color-cta-bg': '#0E639C',
      '--color-cta-bg-hover': '#1177BB',
      '--color-cta-bg-active': '#1682C5',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(14, 99, 156, 0.18)',
      '--color-nav-active-text': '#3794FF',

      '--color-scrollbar-thumb': '#424242',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#181818',
      '--bg-panel': '#1F1F1F',
      '--bg-panel-hover': '#282828',
      '--bg-button': '#282828',
      '--bg-button-hover': '#333333',
      '--bg-input': '#222222',
      '--text-primary': '#CCCCCC',
      '--text-secondary': '#9D9D9D',
      '--text-tertiary': '#6E6E6E',
      '--accent-color': '#3794FF',
      '--border-color': '#383838',
      '--border-light': 'rgba(255, 255, 255, 0.08)',

      '--up-bg-modal': '#1F1F1F',
      '--up-bg-header': '#181818',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.035)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.07)',
      '--up-border': '#383838',
      '--up-border-subtle': '#2B2B2B',
      '--up-text-primary': '#CCCCCC',
      '--up-text-secondary': '#9D9D9D',
      '--up-text-muted': '#6E6E6E',
      '--up-accent': '#0E639C',
      '--up-accent-hover': '#1177BB',
      '--up-accent-subtle': 'rgba(14, 99, 156, 0.15)',
      '--up-accent-border': 'rgba(14, 99, 156, 0.35)',
    },
  },

  // ── 3. Abyss (Deep Ocean Midnight) ──
  {
    id: 'abyss',
    name: 'Abyss',
    category: 'dark',
    isDark: true,
    description: 'Immerse into deep oceanic blues, abyssal midnight waters, and luminous cyan/aquamarine highlights.',
    swatches: ['#000C18', '#051329', '#18A2B8', '#007ACC', '#9CDCFE'],
    particlePalette: [
      { rgb: [0, 150, 255], weight: 0.35 },   // Cool Blue
      { rgb: [24, 162, 184], weight: 0.30 },  // Luminous Cyan
      { rgb: [64, 224, 255], weight: 0.20 },  // Aquamarine
      { rgb: [30, 90, 180], weight: 0.15 },   // Deep Ocean Blue
    ],
    tokens: {
      '--color-bg-base': '#000814',
      '--color-bg-surface': '#041126',
      '--color-bg-elevated': '#081B3A',
      '--color-bg-overlay': '#0C264F',
      '--color-bg-sunken': '#00040A',
      '--color-bg-input': '#061630',
      '--color-bg-surface-glass': 'rgba(4, 17, 38, 0.88)',

      '--color-text-primary': '#E3F2FD',
      '--color-text-secondary': '#82B1FF',
      '--color-text-muted': '#448AFF',
      '--color-text-disabled': '#19335A',
      '--color-text-inverse': '#000814',
      '--color-text-accent': '#00E5FF',

      '--color-border-subtle': '#0A2044',
      '--color-border-default': '#0F2F63',
      '--color-border-strong': '#174794',
      '--color-border-focus': '#00E5FF',
      '--color-border-accent': '#00B0FF',

      '--color-accent-500': '#00B0FF',
      '--color-accent-400': '#00E5FF',
      '--color-cta-bg': '#0091EA',
      '--color-cta-bg-hover': '#00B0FF',
      '--color-cta-bg-active': '#40C4FF',
      '--color-cta-text': '#000814',

      '--color-nav-active-bg': 'rgba(0, 229, 255, 0.14)',
      '--color-nav-active-text': '#00E5FF',

      '--color-scrollbar-thumb': '#0F2F63',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#000814',
      '--bg-panel': '#041126',
      '--bg-panel-hover': '#081B3A',
      '--bg-button': '#081B3A',
      '--bg-button-hover': '#0C264F',
      '--bg-input': '#061630',
      '--text-primary': '#E3F2FD',
      '--text-secondary': '#82B1FF',
      '--text-tertiary': '#448AFF',
      '--accent-color': '#00E5FF',
      '--border-color': '#0F2F63',
      '--border-light': 'rgba(0, 229, 255, 0.1)',

      '--up-bg-modal': '#041126',
      '--up-bg-header': '#000814',
      '--up-bg-surface': 'rgba(0, 229, 255, 0.03)',
      '--up-bg-surface-hover': 'rgba(0, 229, 255, 0.08)',
      '--up-border': '#0F2F63',
      '--up-border-subtle': '#0A2044',
      '--up-text-primary': '#E3F2FD',
      '--up-text-secondary': '#82B1FF',
      '--up-text-muted': '#448AFF',
      '--up-accent': '#00B0FF',
      '--up-accent-hover': '#00E5FF',
      '--up-accent-subtle': 'rgba(0, 229, 255, 0.12)',
      '--up-accent-border': 'rgba(0, 229, 255, 0.3)',
    },
  },

  // ── 4. Monokai (Legendary Sublime & VS Code Charcoal) ──
  {
    id: 'monokai',
    name: 'Monokai',
    category: 'dark',
    isDark: true,
    description: 'The beloved classic with deep charcoal tones, electric lime, fiery yellow, and vivid magenta.',
    swatches: ['#272822', '#1E1F1C', '#A6E22E', '#F92672', '#F8F8F2'],
    particlePalette: [
      { rgb: [174, 129, 255], weight: 0.30 }, // Monokai Violet
      { rgb: [249, 38, 114], weight: 0.30 },  // Vivid Pink
      { rgb: [230, 219, 116], weight: 0.20 }, // Warm Yellow accent
      { rgb: [166, 226, 46], weight: 0.12 },  // Lime
      { rgb: [102, 217, 239], weight: 0.08 }, // Soft Cyan
    ],
    tokens: {
      '--color-bg-base': '#1E1F1C',
      '--color-bg-surface': '#272822',
      '--color-bg-elevated': '#34352E',
      '--color-bg-overlay': '#414339',
      '--color-bg-sunken': '#141512',
      '--color-bg-input': '#1F201C',
      '--color-bg-surface-glass': 'rgba(39, 40, 34, 0.90)',

      '--color-text-primary': '#F8F8F2',
      '--color-text-secondary': '#CFD0C2',
      '--color-text-muted': '#888471',
      '--color-text-disabled': '#49483E',
      '--color-text-inverse': '#1E1F1C',
      '--color-text-accent': '#A6E22E',

      '--color-border-subtle': '#34352E',
      '--color-border-default': '#49483E',
      '--color-border-strong': '#666453',
      '--color-border-focus': '#A6E22E',
      '--color-border-accent': '#F92672',

      '--color-accent-500': '#A6E22E',
      '--color-accent-400': '#B8E94B',
      '--color-cta-bg': '#A6E22E',
      '--color-cta-bg-hover': '#B8E94B',
      '--color-cta-bg-active': '#C9F06A',
      '--color-cta-text': '#141512',

      '--color-nav-active-bg': 'rgba(166, 226, 46, 0.14)',
      '--color-nav-active-text': '#A6E22E',

      '--color-scrollbar-thumb': '#49483E',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#1E1F1C',
      '--bg-panel': '#272822',
      '--bg-panel-hover': '#34352E',
      '--bg-button': '#34352E',
      '--bg-button-hover': '#414339',
      '--bg-input': '#1F201C',
      '--text-primary': '#F8F8F2',
      '--text-secondary': '#CFD0C2',
      '--text-tertiary': '#888471',
      '--accent-color': '#A6E22E',
      '--border-color': '#49483E',
      '--border-light': 'rgba(255, 255, 255, 0.06)',

      '--up-bg-modal': '#272822',
      '--up-bg-header': '#1E1F1C',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.04)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.08)',
      '--up-border': '#49483E',
      '--up-border-subtle': '#34352E',
      '--up-text-primary': '#F8F8F2',
      '--up-text-secondary': '#CFD0C2',
      '--up-text-muted': '#888471',
      '--up-accent': '#A6E22E',
      '--up-accent-hover': '#B8E94B',
      '--up-accent-subtle': 'rgba(166, 226, 46, 0.12)',
      '--up-accent-border': 'rgba(166, 226, 46, 0.3)',
    },
  },

  // ── 5. Monokai Dimmed ──
  {
    id: 'monokai-dimmed',
    name: 'Monokai Dimmed',
    category: 'dark',
    isDark: true,
    description: 'A gentle, low-contrast variation of Monokai designed for marathon late-night studying.',
    swatches: ['#1E1E1E', '#282828', '#9872A2', '#6796E6', '#C5C8C6'],
    particlePalette: [
      { rgb: [152, 114, 162], weight: 0.35 }, // Softer Violet
      { rgb: [205, 120, 150], weight: 0.25 }, // Muted Pink
      { rgb: [103, 150, 230], weight: 0.25 }, // Slate Violet
      { rgb: [180, 160, 120], weight: 0.15 }, // Warm Muted Accent
    ],
    tokens: {
      '--color-bg-base': '#181818',
      '--color-bg-surface': '#202020',
      '--color-bg-elevated': '#2B2B2B',
      '--color-bg-overlay': '#363636',
      '--color-bg-sunken': '#121212',
      '--color-bg-input': '#232323',
      '--color-bg-surface-glass': 'rgba(32, 32, 32, 0.90)',

      '--color-text-primary': '#C5C8C6',
      '--color-text-secondary': '#9B9E9C',
      '--color-text-muted': '#686B69',
      '--color-text-disabled': '#3E413F',
      '--color-text-inverse': '#181818',
      '--color-text-accent': '#9872A2',

      '--color-border-subtle': '#2B2B2B',
      '--color-border-default': '#383838',
      '--color-border-strong': '#4E4E4E',
      '--color-border-focus': '#9872A2',
      '--color-border-accent': '#6796E6',

      '--color-accent-500': '#9872A2',
      '--color-accent-400': '#AB83B5',
      '--color-cta-bg': '#9872A2',
      '--color-cta-bg-hover': '#AB83B5',
      '--color-cta-bg-active': '#BF94C9',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(152, 114, 162, 0.15)',
      '--color-nav-active-text': '#AB83B5',

      '--color-scrollbar-thumb': '#3A3A3A',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#181818',
      '--bg-panel': '#202020',
      '--bg-panel-hover': '#2B2B2B',
      '--bg-button': '#2B2B2B',
      '--bg-button-hover': '#363636',
      '--bg-input': '#232323',
      '--text-primary': '#C5C8C6',
      '--text-secondary': '#9B9E9C',
      '--text-tertiary': '#686B69',
      '--accent-color': '#AB83B5',
      '--border-color': '#383838',
      '--border-light': 'rgba(255, 255, 255, 0.05)',

      '--up-bg-modal': '#202020',
      '--up-bg-header': '#181818',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.03)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.06)',
      '--up-border': '#383838',
      '--up-border-subtle': '#2B2B2B',
      '--up-text-primary': '#C5C8C6',
      '--up-text-secondary': '#9B9E9C',
      '--up-text-muted': '#686B69',
      '--up-accent': '#9872A2',
      '--up-accent-hover': '#AB83B5',
      '--up-accent-subtle': 'rgba(152, 114, 162, 0.12)',
      '--up-accent-border': 'rgba(152, 114, 162, 0.3)',
    },
  },

  // ── 6. Red (Rich Crimson & Burgundy Wine) ──
  {
    id: 'red',
    name: 'Red',
    category: 'dark',
    isDark: true,
    description: 'Rich dark wine surfaces accented with bold ruby crimson and radiant warm amber highlights.',
    swatches: ['#240507', '#34080B', '#E53935', '#FFA436', '#FFEBEE'],
    particlePalette: [
      { rgb: [229, 57, 53], weight: 0.40 },  // Crimson Red
      { rgb: [255, 164, 54], weight: 0.30 }, // Amber Glow
      { rgb: [180, 20, 40], weight: 0.20 },  // Deep Ruby
      { rgb: [255, 110, 110], weight: 0.10 },// Coral
    ],
    tokens: {
      '--color-bg-base': '#1A0406',
      '--color-bg-surface': '#25080A',
      '--color-bg-elevated': '#360E12',
      '--color-bg-overlay': '#4A161B',
      '--color-bg-sunken': '#120204',
      '--color-bg-input': '#2A0B0E',
      '--color-bg-surface-glass': 'rgba(37, 8, 10, 0.90)',

      '--color-text-primary': '#FFEBEF',
      '--color-text-secondary': '#F48FB1',
      '--color-text-muted': '#B75069',
      '--color-text-disabled': '#561922',
      '--color-text-inverse': '#1A0406',
      '--color-text-accent': '#FF5252',

      '--color-border-subtle': '#360E12',
      '--color-border-default': '#4A161B',
      '--color-border-strong': '#682028',
      '--color-border-focus': '#FF5252',
      '--color-border-accent': '#D32F2F',

      '--color-accent-500': '#E53935',
      '--color-accent-400': '#FF5252',
      '--color-cta-bg': '#E53935',
      '--color-cta-bg-hover': '#EF5350',
      '--color-cta-bg-active': '#E57373',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(229, 57, 53, 0.16)',
      '--color-nav-active-text': '#FF5252',

      '--color-scrollbar-thumb': '#4A161B',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#1A0406',
      '--bg-panel': '#25080A',
      '--bg-panel-hover': '#360E12',
      '--bg-button': '#360E12',
      '--bg-button-hover': '#4A161B',
      '--bg-input': '#2A0B0E',
      '--text-primary': '#FFEBEF',
      '--text-secondary': '#F48FB1',
      '--text-tertiary': '#B75069',
      '--accent-color': '#FF5252',
      '--border-color': '#4A161B',
      '--border-light': 'rgba(255, 82, 82, 0.1)',

      '--up-bg-modal': '#25080A',
      '--up-bg-header': '#1A0406',
      '--up-bg-surface': 'rgba(255, 82, 82, 0.04)',
      '--up-bg-surface-hover': 'rgba(255, 82, 82, 0.08)',
      '--up-border': '#4A161B',
      '--up-border-subtle': '#360E12',
      '--up-text-primary': '#FFEBEF',
      '--up-text-secondary': '#F48FB1',
      '--up-text-muted': '#B75069',
      '--up-accent': '#E53935',
      '--up-accent-hover': '#FF5252',
      '--up-accent-subtle': 'rgba(229, 57, 53, 0.12)',
      '--up-accent-border': 'rgba(229, 57, 53, 0.3)',
    },
  },

  // ── 7. Solarized Dark (Ethan Schoonover's Precision Palette) ──
  {
    id: 'solarized-dark',
    name: 'Solarized Dark',
    category: 'dark',
    isDark: true,
    description: 'The mathematically designed teal-blue scheme crafted specifically for visual endurance.',
    swatches: ['#002B36', '#073642', '#2AA198', '#B58900', '#93A1A1'],
    particlePalette: [
      { rgb: [42, 161, 152], weight: 0.35 },  // Teal (Cyan)
      { rgb: [38, 139, 210], weight: 0.30 },  // Blue
      { rgb: [108, 113, 196], weight: 0.20 }, // Muted Violet
      { rgb: [181, 137, 0], weight: 0.15 },   // Solarized Amber
    ],
    tokens: {
      '--color-bg-base': '#00212B',
      '--color-bg-surface': '#002B36',
      '--color-bg-elevated': '#073642',
      '--color-bg-overlay': '#0E4957',
      '--color-bg-sunken': '#00181F',
      '--color-bg-input': '#05313D',
      '--color-bg-surface-glass': 'rgba(0, 43, 54, 0.90)',

      '--color-text-primary': '#93A1A1',
      '--color-text-secondary': '#839496',
      '--color-text-muted': '#586E75',
      '--color-text-disabled': '#0A3F4D',
      '--color-text-inverse': '#00212B',
      '--color-text-accent': '#2AA198',

      '--color-border-subtle': '#073642',
      '--color-border-default': '#0B4858',
      '--color-border-strong': '#135D70',
      '--color-border-focus': '#2AA198',
      '--color-border-accent': '#B58900',

      '--color-accent-500': '#2AA198',
      '--color-accent-400': '#36B5AC',
      '--color-cta-bg': '#2AA198',
      '--color-cta-bg-hover': '#36B5AC',
      '--color-cta-bg-active': '#44C9BF',
      '--color-cta-text': '#00212B',

      '--color-nav-active-bg': 'rgba(42, 161, 152, 0.15)',
      '--color-nav-active-text': '#2AA198',

      '--color-scrollbar-thumb': '#0A4452',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#00212B',
      '--bg-panel': '#002B36',
      '--bg-panel-hover': '#073642',
      '--bg-button': '#073642',
      '--bg-button-hover': '#0E4957',
      '--bg-input': '#05313D',
      '--text-primary': '#93A1A1',
      '--text-secondary': '#839496',
      '--text-tertiary': '#586E75',
      '--accent-color': '#2AA198',
      '--border-color': '#0B4858',
      '--border-light': 'rgba(42, 161, 152, 0.1)',

      '--up-bg-modal': '#002B36',
      '--up-bg-header': '#00212B',
      '--up-bg-surface': 'rgba(42, 161, 152, 0.04)',
      '--up-bg-surface-hover': 'rgba(42, 161, 152, 0.08)',
      '--up-border': '#0B4858',
      '--up-border-subtle': '#073642',
      '--up-text-primary': '#93A1A1',
      '--up-text-secondary': '#839496',
      '--up-text-muted': '#586E75',
      '--up-accent': '#2AA198',
      '--up-accent-hover': '#36B5AC',
      '--up-accent-subtle': 'rgba(42, 161, 152, 0.12)',
      '--up-accent-border': 'rgba(42, 161, 152, 0.3)',
    },
  },

  // ── 8. Tomorrow Night Blue (Midnight Indigo & Sapphire) ──
  {
    id: 'tomorrow-night-blue',
    name: 'Tomorrow Night Blue',
    category: 'dark',
    isDark: true,
    description: 'Crisp midnight indigo tones paired with soft pastel sapphire and warm tangerine highlights.',
    swatches: ['#002451', '#002F68', '#81A2BE', '#DE935F', '#FFFFFF'],
    particlePalette: [
      { rgb: [129, 162, 190], weight: 0.35 }, // Sapphire Blue
      { rgb: [90, 190, 220], weight: 0.30 },  // Cool Cyan
      { rgb: [197, 160, 230], weight: 0.25 }, // Violet
      { rgb: [222, 147, 95], weight: 0.10 },  // Tangerine
    ],
    tokens: {
      '--color-bg-base': '#001D42',
      '--color-bg-surface': '#002451',
      '--color-bg-elevated': '#003370',
      '--color-bg-overlay': '#00428F',
      '--color-bg-sunken': '#001530',
      '--color-bg-input': '#002B5E',
      '--color-bg-surface-glass': 'rgba(0, 36, 81, 0.90)',

      '--color-text-primary': '#FFFFFF',
      '--color-text-secondary': '#C5D4E6',
      '--color-text-muted': '#728CA6',
      '--color-text-disabled': '#28466C',
      '--color-text-inverse': '#001D42',
      '--color-text-accent': '#81A2BE',

      '--color-border-subtle': '#003370',
      '--color-border-default': '#00428F',
      '--color-border-strong': '#0B59B3',
      '--color-border-focus': '#81A2BE',
      '--color-border-accent': '#DE935F',

      '--color-accent-500': '#81A2BE',
      '--color-accent-400': '#9CBAD5',
      '--color-cta-bg': '#81A2BE',
      '--color-cta-bg-hover': '#9CBAD5',
      '--color-cta-bg-active': '#B6CFE6',
      '--color-cta-text': '#001D42',

      '--color-nav-active-bg': 'rgba(129, 162, 190, 0.16)',
      '--color-nav-active-text': '#9CBAD5',

      '--color-scrollbar-thumb': '#00428F',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#001D42',
      '--bg-panel': '#002451',
      '--bg-panel-hover': '#003370',
      '--bg-button': '#003370',
      '--bg-button-hover': '#00428F',
      '--bg-input': '#002B5E',
      '--text-primary': '#FFFFFF',
      '--text-secondary': '#C5D4E6',
      '--text-tertiary': '#728CA6',
      '--accent-color': '#9CBAD5',
      '--border-color': '#00428F',
      '--border-light': 'rgba(255, 255, 255, 0.08)',

      '--up-bg-modal': '#002451',
      '--up-bg-header': '#001D42',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.04)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.08)',
      '--up-border': '#00428F',
      '--up-border-subtle': '#003370',
      '--up-text-primary': '#FFFFFF',
      '--up-text-secondary': '#C5D4E6',
      '--up-text-muted': '#728CA6',
      '--up-accent': '#81A2BE',
      '--up-accent-hover': '#9CBAD5',
      '--up-accent-subtle': 'rgba(129, 162, 190, 0.15)',
      '--up-accent-border': 'rgba(129, 162, 190, 0.35)',
    },
  },

  // ── 9. Visual Studio 2019 Dark (Classic IDE Slate) ──
  {
    id: 'vs2019-dark',
    name: 'Visual Studio 2019 Dark',
    category: 'dark',
    isDark: true,
    description: 'The heavyweight studio dark slate theme with Microsoft purple and electric azure accents.',
    swatches: ['#1E1E1E', '#252526', '#007ACC', '#68217A', '#F1F1F1'],
    particlePalette: [
      { rgb: [0, 122, 204], weight: 0.40 },  // VS Azure
      { rgb: [104, 33, 122], weight: 0.30 }, // Studio Purple
      { rgb: [78, 201, 176], weight: 0.20 }, // Aqua
      { rgb: [220, 220, 220], weight: 0.10 },// Slate
    ],
    tokens: {
      '--color-bg-base': '#1B1B1C',
      '--color-bg-surface': '#252526',
      '--color-bg-elevated': '#2D2D30',
      '--color-bg-overlay': '#3E3E42',
      '--color-bg-sunken': '#161617',
      '--color-bg-input': '#222224',
      '--color-bg-surface-glass': 'rgba(37, 37, 38, 0.90)',

      '--color-text-primary': '#F1F1F1',
      '--color-text-secondary': '#CCCCCC',
      '--color-text-muted': '#858585',
      '--color-text-disabled': '#46464A',
      '--color-text-inverse': '#1B1B1C',
      '--color-text-accent': '#007ACC',

      '--color-border-subtle': '#2D2D30',
      '--color-border-default': '#3F3F46',
      '--color-border-strong': '#52525B',
      '--color-border-focus': '#007ACC',
      '--color-border-accent': '#68217A',

      '--color-accent-500': '#007ACC',
      '--color-accent-400': '#1F9CF0',
      '--color-cta-bg': '#007ACC',
      '--color-cta-bg-hover': '#1F9CF0',
      '--color-cta-bg-active': '#45B0F7',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(0, 122, 204, 0.18)',
      '--color-nav-active-text': '#1F9CF0',

      '--color-scrollbar-thumb': '#424245',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#1B1B1C',
      '--bg-panel': '#252526',
      '--bg-panel-hover': '#2D2D30',
      '--bg-button': '#2D2D30',
      '--bg-button-hover': '#3E3E42',
      '--bg-input': '#222224',
      '--text-primary': '#F1F1F1',
      '--text-secondary': '#CCCCCC',
      '--text-tertiary': '#858585',
      '--accent-color': '#1F9CF0',
      '--border-color': '#3F3F46',
      '--border-light': 'rgba(255, 255, 255, 0.08)',

      '--up-bg-modal': '#252526',
      '--up-bg-header': '#1B1B1C',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.035)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.07)',
      '--up-border': '#3F3F46',
      '--up-border-subtle': '#2D2D30',
      '--up-text-primary': '#F1F1F1',
      '--up-text-secondary': '#CCCCCC',
      '--up-text-muted': '#858585',
      '--up-accent': '#007ACC',
      '--up-accent-hover': '#1F9CF0',
      '--up-accent-subtle': 'rgba(0, 122, 204, 0.15)',
      '--up-accent-border': 'rgba(0, 122, 204, 0.35)',
    },
  },

  // ── 10. Dracula (Legendary Gothic Violet Dark) ──
  {
    id: 'dracula',
    name: 'Dracula',
    category: 'dark',
    isDark: true,
    description: 'A dark theme for vampires with luminous purples, pinks, cyans, and emerald greens.',
    swatches: ['#282A36', '#21222C', '#BD93F9', '#FF79C6', '#F8F8F2'],
    particlePalette: [
      { rgb: [189, 147, 249], weight: 0.35 }, // Dracula Purple
      { rgb: [255, 121, 198], weight: 0.30 }, // Dracula Pink
      { rgb: [139, 233, 253], weight: 0.20 }, // Dracula Cyan
      { rgb: [80, 250, 123], weight: 0.15 },  // Dracula Green
    ],
    tokens: {
      '--color-bg-base': '#1E1F29',
      '--color-bg-surface': '#282A36',
      '--color-bg-elevated': '#343746',
      '--color-bg-overlay': '#44475A',
      '--color-bg-sunken': '#16171E',
      '--color-bg-input': '#232530',
      '--color-bg-surface-glass': 'rgba(40, 42, 54, 0.90)',

      '--color-text-primary': '#F8F8F2',
      '--color-text-secondary': '#D6D6E0',
      '--color-text-muted': '#6272A4',
      '--color-text-disabled': '#383A59',
      '--color-text-inverse': '#1E1F29',
      '--color-text-accent': '#BD93F9',

      '--color-border-subtle': '#343746',
      '--color-border-default': '#44475A',
      '--color-border-strong': '#6272A4',
      '--color-border-focus': '#BD93F9',
      '--color-border-accent': '#FF79C6',

      '--color-accent-500': '#BD93F9',
      '--color-accent-400': '#CAA6FB',
      '--color-cta-bg': '#BD93F9',
      '--color-cta-bg-hover': '#CAA6FB',
      '--color-cta-bg-active': '#D9B9FD',
      '--color-cta-text': '#1E1F29',

      '--color-nav-active-bg': 'rgba(189, 147, 249, 0.15)',
      '--color-nav-active-text': '#CAA6FB',

      '--color-scrollbar-thumb': '#44475A',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#1E1F29',
      '--bg-panel': '#282A36',
      '--bg-panel-hover': '#343746',
      '--bg-button': '#343746',
      '--bg-button-hover': '#44475A',
      '--bg-input': '#232530',
      '--text-primary': '#F8F8F2',
      '--text-secondary': '#D6D6E0',
      '--text-tertiary': '#6272A4',
      '--accent-color': '#CAA6FB',
      '--border-color': '#44475A',
      '--border-light': 'rgba(255, 255, 255, 0.08)',

      '--up-bg-modal': '#282A36',
      '--up-bg-header': '#1E1F29',
      '--up-bg-surface': 'rgba(255, 255, 255, 0.04)',
      '--up-bg-surface-hover': 'rgba(255, 255, 255, 0.08)',
      '--up-border': '#44475A',
      '--up-border-subtle': '#343746',
      '--up-text-primary': '#F8F8F2',
      '--up-text-secondary': '#D6D6E0',
      '--up-text-muted': '#6272A4',
      '--up-accent': '#BD93F9',
      '--up-accent-hover': '#CAA6FB',
      '--up-accent-subtle': 'rgba(189, 147, 249, 0.15)',
      '--up-accent-border': 'rgba(189, 147, 249, 0.35)',
    },
  },

  // ── 11. Light Modern (Clean Studio Light) ──
  {
    id: 'light-modern',
    name: 'Light Modern',
    category: 'light',
    isDark: false,
    description: 'Crisp daytime aesthetic with clean light-gray surfaces, high legibility, and refined cobalt blue.',
    swatches: ['#F3F4F6', '#FFFFFF', '#005FB8', '#007ACC', '#1F2937'],
    particlePalette: [
      { rgb: [37, 99, 235], weight: 0.35 },   // Soft Blue
      { rgb: [124, 58, 237], weight: 0.25 },  // Violet
      { rgb: [219, 39, 119], weight: 0.20 },  // Soft Pink
      { rgb: [79, 70, 229], weight: 0.20 },   // Indigo
    ],
    tokens: {
      '--color-bg-base': '#F3F4F6',
      '--color-bg-surface': '#FFFFFF',
      '--color-bg-elevated': '#F8F9FA',
      '--color-bg-overlay': '#E5E7EB',
      '--color-bg-sunken': '#E5E7EB',
      '--color-bg-input': '#FFFFFF',
      '--color-bg-surface-glass': 'rgba(255, 255, 255, 0.92)',

      '--color-text-primary': '#111827',
      '--color-text-secondary': '#374151',
      '--color-text-muted': '#6B7280',
      '--color-text-disabled': '#9CA3AF',
      '--color-text-inverse': '#FFFFFF',
      '--color-text-accent': '#005FB8',

      '--color-border-subtle': '#E5E7EB',
      '--color-border-default': '#D1D5DB',
      '--color-border-strong': '#9CA3AF',
      '--color-border-focus': '#005FB8',
      '--color-border-accent': '#2563EB',

      '--color-accent-500': '#005FB8',
      '--color-accent-400': '#1A7FD8',
      '--color-cta-bg': '#005FB8',
      '--color-cta-bg-hover': '#004C94',
      '--color-cta-bg-active': '#003970',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(0, 95, 184, 0.10)',
      '--color-nav-active-text': '#005FB8',

      '--color-scrollbar-thumb': '#C4C8D0',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#F3F4F6',
      '--bg-panel': '#FFFFFF',
      '--bg-panel-hover': '#F8F9FA',
      '--bg-button': '#F3F4F6',
      '--bg-button-hover': '#E5E7EB',
      '--bg-input': '#FFFFFF',
      '--text-primary': '#111827',
      '--text-secondary': '#374151',
      '--text-tertiary': '#6B7280',
      '--accent-color': '#005FB8',
      '--border-color': '#D1D5DB',
      '--border-light': 'rgba(0, 0, 0, 0.08)',

      '--up-bg-modal': '#FFFFFF',
      '--up-bg-header': '#F8F9FA',
      '--up-bg-surface': 'rgba(0, 0, 0, 0.03)',
      '--up-bg-surface-hover': 'rgba(0, 0, 0, 0.06)',
      '--up-border': '#E5E7EB',
      '--up-border-subtle': '#F3F4F6',
      '--up-text-primary': '#111827',
      '--up-text-secondary': '#374151',
      '--up-text-muted': '#6B7280',
      '--up-accent': '#005FB8',
      '--up-accent-hover': '#004C94',
      '--up-accent-subtle': 'rgba(0, 95, 184, 0.08)',
      '--up-accent-border': 'rgba(0, 95, 184, 0.25)',
    },
  },

  // ── 12. Light+ (Classic VS Code Light+) ──
  {
    id: 'light-plus',
    name: 'Light+',
    category: 'light',
    isDark: false,
    description: 'The standard Visual Studio Code light theme with white canvas and crisp editor accents.',
    swatches: ['#FFFFFF', '#F3F3F3', '#007ACC', '#008000', '#24292E'],
    particlePalette: [
      { rgb: [0, 122, 204], weight: 0.35 },   // Soft Blue
      { rgb: [139, 92, 246], weight: 0.25 },  // Violet
      { rgb: [236, 72, 153], weight: 0.20 },  // Soft Pink
      { rgb: [16, 185, 129], weight: 0.20 },  // Soft Emerald
    ],
    tokens: {
      '--color-bg-base': '#F8F8F8',
      '--color-bg-surface': '#FFFFFF',
      '--color-bg-elevated': '#F3F3F3',
      '--color-bg-overlay': '#E8E8E8',
      '--color-bg-sunken': '#E4E4E4',
      '--color-bg-input': '#FFFFFF',
      '--color-bg-surface-glass': 'rgba(255, 255, 255, 0.94)',

      '--color-text-primary': '#24292E',
      '--color-text-secondary': '#444D56',
      '--color-text-muted': '#6A737D',
      '--color-text-disabled': '#959DA5',
      '--color-text-inverse': '#FFFFFF',
      '--color-text-accent': '#007ACC',

      '--color-border-subtle': '#E8E8E8',
      '--color-border-default': '#D0D0D0',
      '--color-border-strong': '#999999',
      '--color-border-focus': '#007ACC',
      '--color-border-accent': '#0E639C',

      '--color-accent-500': '#007ACC',
      '--color-accent-400': '#1A8CD8',
      '--color-cta-bg': '#007ACC',
      '--color-cta-bg-hover': '#0062A3',
      '--color-cta-bg-active': '#004C80',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(0, 122, 204, 0.10)',
      '--color-nav-active-text': '#007ACC',

      '--color-scrollbar-thumb': '#C8C8C8',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#F8F8F8',
      '--bg-panel': '#FFFFFF',
      '--bg-panel-hover': '#F3F3F3',
      '--bg-button': '#F3F3F3',
      '--bg-button-hover': '#E8E8E8',
      '--bg-input': '#FFFFFF',
      '--text-primary': '#24292E',
      '--text-secondary': '#444D56',
      '--text-tertiary': '#6A737D',
      '--accent-color': '#007ACC',
      '--border-color': '#D0D0D0',
      '--border-light': 'rgba(0, 0, 0, 0.08)',

      '--up-bg-modal': '#FFFFFF',
      '--up-bg-header': '#F3F3F3',
      '--up-bg-surface': 'rgba(0, 0, 0, 0.03)',
      '--up-bg-surface-hover': 'rgba(0, 0, 0, 0.06)',
      '--up-border': '#D0D0D0',
      '--up-border-subtle': '#E8E8E8',
      '--up-text-primary': '#24292E',
      '--up-text-secondary': '#444D56',
      '--up-text-muted': '#6A737D',
      '--up-accent': '#007ACC',
      '--up-accent-hover': '#0062A3',
      '--up-accent-subtle': 'rgba(0, 122, 204, 0.08)',
      '--up-accent-border': 'rgba(0, 122, 204, 0.25)',
    },
  },

  // ── 13. Solarized Light (Warm Manuscript Parchment) ──
  {
    id: 'solarized-light',
    name: 'Solarized Light',
    category: 'light',
    isDark: false,
    description: 'Warm, low-strain parchment and cream tones with rich amber, olive, and teal accents.',
    swatches: ['#FDF6E3', '#EEE8D5', '#B58900', '#2AA198', '#657B83'],
    particlePalette: [
      { rgb: [42, 161, 152], weight: 0.35 },  // Muted Teal
      { rgb: [38, 139, 210], weight: 0.30 },  // Blue
      { rgb: [181, 137, 0], weight: 0.20 },   // Warm Amber
      { rgb: [133, 153, 0], weight: 0.15 },   // Olive
    ],
    tokens: {
      '--color-bg-base': '#F5ECCF',
      '--color-bg-surface': '#FDF6E3',
      '--color-bg-elevated': '#EEE8D5',
      '--color-bg-overlay': '#E0D8C3',
      '--color-bg-sunken': '#D8CFB8',
      '--color-bg-input': '#FFFFFF',
      '--color-bg-surface-glass': 'rgba(253, 246, 227, 0.94)',

      '--color-text-primary': '#073642',
      '--color-text-secondary': '#586E75',
      '--color-text-muted': '#839496',
      '--color-text-disabled': '#93A1A1',
      '--color-text-inverse': '#FDF6E3',
      '--color-text-accent': '#B58900',

      '--color-border-subtle': '#EEE8D5',
      '--color-border-default': '#E0D8C3',
      '--color-border-strong': '#93A1A1',
      '--color-border-focus': '#B58900',
      '--color-border-accent': '#2AA198',

      '--color-accent-500': '#B58900',
      '--color-accent-400': '#CB9B02',
      '--color-cta-bg': '#B58900',
      '--color-cta-bg-hover': '#987300',
      '--color-cta-bg-active': '#7D5E00',
      '--color-cta-text': '#FFFFFF',

      '--color-nav-active-bg': 'rgba(181, 137, 0, 0.12)',
      '--color-nav-active-text': '#B58900',

      '--color-scrollbar-thumb': '#D0C8B0',
      '--color-scrollbar-track': 'transparent',

      '--bg-main': '#F5ECCF',
      '--bg-panel': '#FDF6E3',
      '--bg-panel-hover': '#EEE8D5',
      '--bg-button': '#EEE8D5',
      '--bg-button-hover': '#E0D8C3',
      '--bg-input': '#FFFFFF',
      '--text-primary': '#073642',
      '--text-secondary': '#586E75',
      '--text-tertiary': '#839496',
      '--accent-color': '#B58900',
      '--border-color': '#E0D8C3',
      '--border-light': 'rgba(0, 0, 0, 0.08)',

      '--up-bg-modal': '#FDF6E3',
      '--up-bg-header': '#EEE8D5',
      '--up-bg-surface': 'rgba(0, 0, 0, 0.035)',
      '--up-bg-surface-hover': 'rgba(0, 0, 0, 0.07)',
      '--up-border': '#E0D8C3',
      '--up-border-subtle': '#EEE8D5',
      '--up-text-primary': '#073642',
      '--up-text-secondary': '#586E75',
      '--up-text-muted': '#839496',
      '--up-accent': '#B58900',
      '--up-accent-hover': '#987300',
      '--up-accent-subtle': 'rgba(181, 137, 0, 0.10)',
      '--up-accent-border': 'rgba(181, 137, 0, 0.28)',
    },
  },
];

/**
 * Find theme definition by ID.
 */
export function getThemeById(themeId: string): ThemeDefinition {
  const found = THEMES.find(t => t.id === themeId);
  return found || THEMES[1]; // default to Dark Modern
}

/**
 * Resolve 'system' to an actual dark or light theme definition.
 */
export function resolveTheme(themeId: string, systemIsDark: boolean = true): ThemeDefinition {
  if (themeId === 'system') {
    const targetId = systemIsDark ? 'dark-modern' : 'light-modern';
    const resolved = getThemeById(targetId);
    return {
      ...resolved,
      id: 'system',
      name: `System Default (${systemIsDark ? 'Dark' : 'Light'})`,
      category: 'system',
      isDark: systemIsDark,
    };
  }
  return getThemeById(themeId);
}

/**
 * Apply CSS custom properties to the document root with smooth transition management.
 */
export function applyThemeTokens(themeDef: ThemeDefinition, withTransition: boolean = true): void {
  if (typeof document === 'undefined') return;

  const root = document.documentElement;

  // Add transition class if enabled and motion is not reduced
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (withTransition && !prefersReducedMotion) {
    root.classList.add('theme-transitioning');
  }

  // Set theme data attributes
  root.setAttribute('data-theme', themeDef.id);
  root.setAttribute('data-theme-category', themeDef.isDark ? 'dark' : 'light');

  // Apply all token properties defined in theme
  for (const [key, val] of Object.entries(themeDef.tokens)) {
    if (val !== undefined) {
      root.style.setProperty(key, val);
    }
  }

  // Ensure all standardized workspace tokens are explicitly set on root
  const t = themeDef.tokens;
  const standardTokens: Record<string, string> = {
    '--app-bg': t['--app-bg'] || t['--color-bg-base'] || t['--bg-main'],
    '--surface-primary': t['--surface-primary'] || t['--color-bg-surface'] || t['--bg-panel'],
    '--surface-secondary': t['--surface-secondary'] || t['--color-bg-elevated'] || t['--bg-button'],
    '--surface-elevated': t['--surface-elevated'] || t['--color-bg-elevated'] || t['--bg-button'],
    '--surface-input': t['--surface-input'] || t['--color-bg-input'] || t['--bg-input'],
    '--text-primary': t['--text-primary'] || t['--color-text-primary'] || (themeDef.isDark ? '#F0F3F8' : '#111827'),
    '--text-secondary': t['--text-secondary'] || t['--color-text-secondary'] || (themeDef.isDark ? '#B8C2D6' : '#4B5563'),
    '--text-muted': t['--text-muted'] || t['--color-text-muted'] || t['--text-tertiary'] || (themeDef.isDark ? '#687288' : '#9CA3AF'),
    '--border-default': t['--border-default'] || t['--color-border-default'] || t['--border-color'] || (themeDef.isDark ? '#232A3E' : '#D1D5DB'),
    '--border-subtle': t['--border-subtle'] || t['--color-border-subtle'] || (themeDef.isDark ? '#191E2E' : '#E5E7EB'),
    '--accent-primary': t['--accent-primary'] || t['--color-accent-500'] || t['--accent-color'],
    '--accent-hover': t['--accent-hover'] || t['--color-accent-400'] || t['--accent-color'],
    '--accent-contrast': t['--accent-contrast'] || t['--color-cta-text'] || (themeDef.isDark ? '#000000' : '#FFFFFF'),
    '--button-primary-bg': t['--button-primary-bg'] || t['--color-cta-bg'] || t['--color-accent-500'],
    '--button-primary-text': t['--button-primary-text'] || t['--color-cta-text'] || (themeDef.isDark ? '#000000' : '#FFFFFF'),
    '--button-secondary-bg': t['--button-secondary-bg'] || t['--color-bg-elevated'] || t['--bg-button'],
    '--hover-bg': t['--hover-bg'] || t['--color-bg-overlay'] || t['--bg-panel-hover'],
    '--selected-bg': t['--selected-bg'] || t['--color-nav-active-bg'] || (themeDef.isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)'),
    '--focus-ring': t['--focus-ring'] || t['--color-border-focus'] || t['--accent-color'],
    '--success': t['--success'] || '#16A34A',
    '--warning': t['--warning'] || '#D97706',
    '--danger': t['--danger'] || '#DC2626',
    '--chat-user-bg': t['--chat-user-bg'] || t['--color-bg-elevated'] || t['--bg-button'],
    '--chat-assistant-bg': t['--chat-assistant-bg'] || 'transparent',
    '--chat-composer-bg': t['--chat-composer-bg'] || t['--color-bg-input'] || t['--bg-input'],
    '--source-card-bg': t['--source-card-bg'] || t['--color-bg-elevated'] || t['--bg-button'],
    '--sidebar-bg': t['--sidebar-bg'] || t['--color-bg-surface'] || t['--bg-panel'],
    '--scrollbar-thumb': t['--scrollbar-thumb'] || t['--color-scrollbar-thumb'] || (themeDef.isDark ? '#3A3A3A' : '#C4C8D0'),
    '--particle-color-1': t['--particle-color-1'] || (themeDef.particlePalette?.[0] ? `rgb(${themeDef.particlePalette[0].rgb.join(', ')})` : 'rgb(59, 130, 246)'),
    '--particle-color-2': t['--particle-color-2'] || (themeDef.particlePalette?.[1] ? `rgb(${themeDef.particlePalette[1].rgb.join(', ')})` : 'rgb(99, 102, 241)'),
    '--particle-color-3': t['--particle-color-3'] || (themeDef.particlePalette?.[2] ? `rgb(${themeDef.particlePalette[2].rgb.join(', ')})` : 'rgb(139, 92, 246)'),
    '--particle-color-4': t['--particle-color-4'] || (themeDef.particlePalette?.[3] ? `rgb(${themeDef.particlePalette[3].rgb.join(', ')})` : 'rgb(236, 72, 153)'),
    '--particle-opacity': t['--particle-opacity'] || (themeDef.isDark ? '0.55' : '0.40'),
    '--particle-glow': t['--particle-glow'] || (themeDef.isDark ? '0.22' : '0.12'),
  };

  for (const [key, val] of Object.entries(standardTokens)) {
    if (val) {
      root.style.setProperty(key, val);
    }
  }

  // Remove transition class after 280ms
  if (withTransition && !prefersReducedMotion) {
    setTimeout(() => {
      root.classList.remove('theme-transitioning');
    }, 280);
  }
}
