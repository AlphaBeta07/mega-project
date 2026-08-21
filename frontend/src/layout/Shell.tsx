/**
 * Shell.tsx — Base Layout Shell
 *
 * Provides the top-level page structure for StudySnap AI:
 *   - Full-viewport dark background (--color-bg-base)
 *   - Optional fixed header slot (--header-height: 60px)
 *   - Main content area that fills the remaining viewport
 *   - A responsive page container with max-width capping
 *
 * Usage:
 *   <Shell header={<Navbar />}>
 *     <YourPageContent />
 *   </Shell>
 *
 * Props:
 *   header?      — Any React node rendered in the fixed top bar slot
 *   children     — The main page content (fills remaining height)
 *   maxWidth?    — Override container max-width token
 *                  'content' = 860px (for chat/note reading views)
 *                  'full'    = 1400px (for wide app layout — default)
 *                  'none'    = no max-width constraint
 *   padded?      — Add horizontal padding to the content area (default: false)
 *                  Use false when a sub-layout handles its own padding (e.g. sidebar grids)
 *   className?   — Additional class on the inner content wrapper
 */

import type { ReactNode, CSSProperties } from 'react';

type ContainerWidth = 'full' | 'content' | 'none';

interface ShellProps {
  header?:    ReactNode;
  children:   ReactNode;
  maxWidth?:  ContainerWidth;
  padded?:    boolean;
  className?: string;
  style?:     CSSProperties;
}

const widthMap: Record<ContainerWidth, string> = {
  full:    'var(--container-max)',
  content: 'var(--container-content)',
  none:    'none',
};

export function Shell({
  header,
  children,
  maxWidth = 'full',
  padded   = false,
  className = '',
  style,
}: ShellProps) {
  return (
    <div className="ss-shell" style={style}>
      {/* ── Header Slot ─────────────────────────────────────────── */}
      {header && (
        <div className="ss-shell__header" role="banner">
          {header}
        </div>
      )}

      {/* ── Main Area ───────────────────────────────────────────── */}
      <main
        className={`ss-shell__main ${className}`.trim()}
        style={{
          /* Offset for fixed header height when header is provided */
          marginTop: header ? 'var(--header-height)' : undefined,
        }}
      >
        <div
          className="ss-shell__container"
          style={{
            maxWidth: widthMap[maxWidth],
            padding: padded ? '0 var(--space-6)' : undefined,
          }}
        >
          {children}
        </div>
      </main>
    </div>
  );
}

/*
  ── Responsive Breakpoints (reference) ────────────────────────
  These match the tokens defined in tokens.css. Use them in CSS
  or inline logic:

  --breakpoint-sm:  640px  → single-column mobile layout
  --breakpoint-md:  768px  → tablet; collapse sidebars
  --breakpoint-lg: 1024px  → show left sidebar
  --breakpoint-xl: 1280px  → show both sidebars (full layout)

  The CSS rules for .ss-shell are in shell.css (imported below
  when you add it to your CSS pipeline, or placed directly in
  index.css). The token names above are what the CSS uses.
  ──────────────────────────────────────────────────────────── */
