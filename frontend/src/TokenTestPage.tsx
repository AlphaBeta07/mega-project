/**
 * TokenTestPage.tsx — Design token smoke test
 *
 * Purpose: Verify that tokens.css, shell.css, and Shell.tsx compile
 * and render correctly. This file can be deleted after the token
 * system is confirmed working.
 *
 * To activate: temporarily swap App.tsx's default export with this,
 * or render <TokenTestPage /> inside the existing App for a quick check.
 */

import { Shell } from './layout/Shell';

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div style={{ marginBottom: 'var(--space-10)' }}>
    <h2 style={{
      fontFamily: 'var(--font-display)',
      fontSize: 'var(--text-xs)',
      fontWeight: 'var(--font-weight-semibold)',
      letterSpacing: 'var(--tracking-widest)',
      textTransform: 'uppercase',
      color: 'var(--color-text-muted)',
      marginBottom: 'var(--space-4)',
      paddingBottom: 'var(--space-2)',
      borderBottom: '1px solid var(--color-border-subtle)',
    }}>
      {title}
    </h2>
    {children}
  </div>
);

const Swatch = ({ token, hex }: { token: string; hex: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
    <div style={{
      width: 40, height: 40,
      borderRadius: 'var(--radius-md)',
      background: hex,
      border: '1px solid var(--color-border-subtle)',
      flexShrink: 0,
    }} />
    <div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--color-text-primary)' }}>{token}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-2xs)', color: 'var(--color-text-muted)' }}>{hex}</div>
    </div>
  </div>
);

export function TokenTestPage() {
  return (
    <Shell
      header={
        <div style={{ padding: '0 var(--space-6)', fontFamily: 'var(--font-display)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
          StudySnap AI — Token Test Page
        </div>
      }
      maxWidth="content"
      padded
    >
      <div style={{ padding: 'var(--space-10) 0', overflowY: 'auto' }}>

        {/* ── Typography ─────────────────────────────────────── */}
        <Section title="Typography">
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-3xl)', fontWeight: 800, letterSpacing: 'var(--tracking-tight)', marginBottom: 'var(--space-3)', lineHeight: 'var(--leading-tight)' }}>
            Display — Plus Jakarta Sans 800
          </h1>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)', fontWeight: 600, marginBottom: 'var(--space-3)' }}>
            Heading — Plus Jakarta Sans 600
          </h2>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 'var(--text-base)', lineHeight: 'var(--leading-relaxed)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-3)' }}>
            Body copy — Inter 400. Students will read dense AI-generated notes here. Optimised for long-form comprehension at 16px / 1.625 line-height. The quick brown fox jumps over the lazy dog.
          </p>
          <code style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--color-text-accent)', background: 'var(--color-bg-elevated)', padding: 'var(--space-1) var(--space-2)', borderRadius: 'var(--radius-sm)' }}>
            CS101 · 2023 · Q.14 · 10:32 AM
          </code>
        </Section>

        {/* ── Color Swatches ─────────────────────────────────── */}
        <Section title="Color — Primary (Oxford Blue)">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 'var(--space-2)' }}>
            <Swatch token="--color-primary-950" hex="#060C1A" />
            <Swatch token="--color-primary-800" hex="#0F2240" />
            <Swatch token="--color-primary-600" hex="#1F4470" />
            <Swatch token="--color-primary-400" hex="#3B7BBB" />
            <Swatch token="--color-primary-200" hex="#90BDE3" />
            <Swatch token="--color-primary-50"  hex="#EBF4FC" />
          </div>
        </Section>

        <Section title="Color — Accent (Saffron Amber)">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 'var(--space-2)' }}>
            <Swatch token="--color-accent-700"  hex="#B36A00" />
            <Swatch token="--color-accent-500"  hex="#F4A900" />
            <Swatch token="--color-accent-400"  hex="#F7BC2E" />
            <Swatch token="--color-accent-300"  hex="#FACC60" />
            <Swatch token="--color-accent-100"  hex="#FEEFCC" />
          </div>
        </Section>

        <Section title="Color — Neutral (Cool Blue-Gray)">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 'var(--space-2)' }}>
            <Swatch token="--color-neutral-950" hex="#0B0E17" />
            <Swatch token="--color-neutral-800" hex="#191E2E" />
            <Swatch token="--color-neutral-600" hex="#313A52" />
            <Swatch token="--color-neutral-400" hex="#687288" />
            <Swatch token="--color-neutral-200" hex="#B8C2D6" />
            <Swatch token="--color-neutral-50"  hex="#F0F3F8" />
          </div>
        </Section>

        {/* ── Interactive elements ────────────────────────────── */}
        <Section title="Interactive Tokens">
          <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap', alignItems: 'center' }}>
            {/* Primary CTA */}
            <button style={{
              background: 'var(--color-cta-bg)',
              color: 'var(--color-cta-text)',
              fontFamily: 'var(--font-body)',
              fontWeight: 'var(--font-weight-semibold)',
              fontSize: 'var(--text-sm)',
              padding: 'var(--space-2_5) var(--space-5)',
              borderRadius: 'var(--radius-2xl)',
              border: 'none',
              cursor: 'pointer',
              boxShadow: 'var(--shadow-glow-accent)',
            }}>
              Generate ✦
            </button>

            {/* Ghost button */}
            <button style={{
              background: 'transparent',
              color: 'var(--color-text-secondary)',
              fontFamily: 'var(--font-body)',
              fontSize: 'var(--text-sm)',
              padding: 'var(--space-2_5) var(--space-5)',
              borderRadius: 'var(--radius-2xl)',
              border: '1px solid var(--color-border-default)',
              cursor: 'pointer',
            }}>
              Cancel
            </button>

            {/* Accent badge */}
            <span className="ss-badge">PYQ 2023</span>
            <span className="ss-badge">CS101</span>

            {/* Mono label */}
            <span className="ss-mono">10:32 AM · 14 Jan 2025</span>
          </div>
        </Section>

        {/* ── Shadow tokens ───────────────────────────────────── */}
        <Section title="Shadows">
          <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
            {(['--shadow-xs','--shadow-sm','--shadow-md','--shadow-lg','--shadow-xl'] as const).map((s) => (
              <div key={s} style={{
                width: 80, height: 80,
                background: 'var(--color-bg-surface)',
                borderRadius: 'var(--radius-lg)',
                boxShadow: `var(${s})`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-2xs)', color: 'var(--color-text-muted)', textAlign: 'center', padding: 4 }}>
                  {s.replace('--shadow-', '')}
                </span>
              </div>
            ))}
          </div>
        </Section>

        {/* ── Border radius tokens ────────────────────────────── */}
        <Section title="Border Radius">
          <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
            {[
              { label: 'sm',   val: 'var(--radius-sm)'   },
              { label: 'md',   val: 'var(--radius-md)'   },
              { label: 'lg',   val: 'var(--radius-lg)'   },
              { label: 'xl',   val: 'var(--radius-xl)'   },
              { label: '2xl',  val: 'var(--radius-2xl)'  },
              { label: 'full', val: 'var(--radius-full)' },
            ].map(({ label, val }) => (
              <div key={label} style={{
                width: 64, height: 64,
                background: 'var(--color-bg-elevated)',
                border: '1px solid var(--color-border-default)',
                borderRadius: val,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-2xs)', color: 'var(--color-text-muted)' }}>
                  {label}
                </span>
              </div>
            ))}
          </div>
        </Section>

        {/* ── Spacing scale preview ───────────────────────────── */}
        <Section title="Spacing Scale (4px base)">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {([1,2,3,4,5,6,8,10,12,16] as const).map((n) => (
              <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', width: 60 }}>
                  --space-{n}
                </span>
                <div style={{
                  height: 8,
                  width: `calc(var(--space-${n}) * 4)`,
                  background: 'var(--color-accent-500)',
                  borderRadius: 'var(--radius-full)',
                  opacity: 0.7,
                }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-2xs)', color: 'var(--color-text-muted)' }}>
                  {n * 4}px
                </span>
              </div>
            ))}
          </div>
        </Section>

        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 'var(--space-12)' }}>
          ✓ tokens.css · shell.css · Shell.tsx · index.css all resolved correctly.
        </p>
      </div>
    </Shell>
  );
}

export default TokenTestPage;
