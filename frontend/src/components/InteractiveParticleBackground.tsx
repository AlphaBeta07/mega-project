import React, { useEffect, useRef } from 'react';
import { useTheme } from '../context/ThemeContext';
import './InteractiveParticleBackground.css';

export interface ParticleConfig {
  particleCount?: number;
  mobileParticleCount?: number;
  interactionRadius?: number;
  cursorForce?: number;
  springStrength?: number;
  friction?: number;
  ambientSpeed?: number;
  baseOpacity?: number;
}

interface Particle {
  relX: number;
  relY: number;
  originX: number;
  originY: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseRadius: number;
  currentRadius: number;
  // Current animated RGB
  rgb: [number, number, number];
  // Target theme RGB for smooth color morphing
  targetRgb: [number, number, number];
  baseAlpha: number;
  targetBaseAlpha: number;
  currentAlpha: number;
  phaseX: number;
  phaseY: number;
  speed: number;
  ambientRadius: number;
  isAccent: boolean;
}

// Fallback Google Antigravity palette (blue, violet, purple, pink)
const DEFAULT_COLOR_PALETTE: { rgb: [number, number, number]; weight: number }[] = [
  { rgb: [59, 130, 246], weight: 0.35 },  // Electric Blue
  { rgb: [99, 102, 241], weight: 0.25 },  // Indigo
  { rgb: [139, 92, 246], weight: 0.20 },  // Violet
  { rgb: [168, 85, 247], weight: 0.12 },  // Purple
  { rgb: [236, 72, 153], weight: 0.08 },  // Soft Pink / Fuchsia
];

function parseCssColor(colorStr: string): [number, number, number] | null {
  if (!colorStr) return null;
  const trimmed = colorStr.trim();
  const rgbMatch = trimmed.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
  if (rgbMatch) {
    return [parseInt(rgbMatch[1], 10), parseInt(rgbMatch[2], 10), parseInt(rgbMatch[3], 10)];
  }
  if (trimmed.startsWith('#')) {
    let hex = trimmed.slice(1);
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    if (hex.length === 6) {
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
      ];
    }
  }
  return null;
}

function getEffectivePalette(resolvedTheme: any): { rgb: [number, number, number]; weight: number }[] {
  if (resolvedTheme?.particlePalette?.length) {
    return resolvedTheme.particlePalette;
  }
  // Fallback: read from injected CSS variables on document.documentElement
  if (typeof document !== 'undefined') {
    const rootStyle = getComputedStyle(document.documentElement);
    const c1 = parseCssColor(rootStyle.getPropertyValue('--particle-color-1'));
    const c2 = parseCssColor(rootStyle.getPropertyValue('--particle-color-2'));
    const c3 = parseCssColor(rootStyle.getPropertyValue('--particle-color-3'));
    const c4 = parseCssColor(rootStyle.getPropertyValue('--particle-color-4'));
    if (c1 && c2) {
      return [
        { rgb: c1, weight: 0.35 },
        { rgb: c2, weight: 0.25 },
        { rgb: c3 || c2, weight: 0.20 },
        { rgb: c4 || c1, weight: 0.20 },
      ];
    }
  }
  return DEFAULT_COLOR_PALETTE;
}

function pickPaletteColor(palette: { rgb: [number, number, number]; weight: number }[]): [number, number, number] {
  if (!palette || palette.length === 0) return DEFAULT_COLOR_PALETTE[0].rgb;
  const r = Math.random();
  let accumulated = 0;
  for (const item of palette) {
    accumulated += item.weight;
    if (r <= accumulated) {
      return [...item.rgb];
    }
  }
  return [...palette[0].rgb];
}

export const InteractiveParticleBackground: React.FC<ParticleConfig> = ({
  particleCount = 320,
  mobileParticleCount = 110,
  interactionRadius = 160,
  cursorForce = 0.09,
  springStrength = 0.045,
  friction = 0.88,
  ambientSpeed = 0.0012,
  baseOpacity = 0.55,
}) => {
  const { resolvedTheme } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const currentThemeRef = useRef(resolvedTheme);
  currentThemeRef.current = resolvedTheme;

  // Pointer state tracked via ref (never causes React re-renders)
  const pointerRef = useRef({
    x: -9999,
    y: -9999,
    prevX: -9999,
    prevY: -9999,
    vx: 0,
    vy: 0,
    active: false,
    lastActiveTime: 0,
  });

  // Dynamically update target particle colors and opacity whenever theme changes
  useEffect(() => {
    if (!particlesRef.current.length) return;
    const currentPalette = getEffectivePalette(resolvedTheme);
    const isLight = !resolvedTheme?.isDark;
    const effectiveBaseOpacity = isLight ? baseOpacity * 1.35 : baseOpacity;

    particlesRef.current.forEach(p => {
      p.targetRgb = pickPaletteColor(currentPalette);
      p.targetBaseAlpha = (0.28 + Math.random() * 0.42) * effectiveBaseOpacity;
    });

    // In reduced-motion mode, redraw static dots immediately with new theme colors
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
          particlesRef.current.forEach(p => {
            p.rgb = [...p.targetRgb];
            p.baseAlpha = p.targetBaseAlpha;
            ctx.fillStyle = `rgba(${p.rgb[0]}, ${p.rgb[1]}, ${p.rgb[2]}, ${p.baseAlpha * 0.7})`;
            ctx.beginPath();
            ctx.arc(p.originX, p.originY, p.baseRadius, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
    }
  }, [resolvedTheme, baseOpacity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Determine particle count based on viewport
    const count = width <= 640 ? mobileParticleCount : (width <= 1024 ? Math.round(particleCount * 0.65) : particleCount);

    const initialPalette = getEffectivePalette(currentThemeRef.current);
    const isLight = !currentThemeRef.current?.isDark;
    const effectiveBaseOpacity = isLight ? baseOpacity * 1.35 : baseOpacity;

    // Initialize particles with organic distribution and normalized coordinates
    const particles: Particle[] = [];

    for (let i = 0; i < count; i++) {
      const relX = Math.random();
      const relY = Math.random();
      const originX = relX * width;
      const originY = relY * height;

      // Varied dot size distribution (mostly tiny, some medium, rare accent)
      const sizeRand = Math.random();
      let baseRadius = 1.0 + Math.random() * 0.6; // 65% tiny dots ~1.0-1.6px
      let isAccent = false;

      if (sizeRand > 0.88) {
        baseRadius = 2.0 + Math.random() * 0.6; // Accent dots ~2.0-2.6px
        isAccent = true;
      } else if (sizeRand > 0.65) {
        baseRadius = 1.5 + Math.random() * 0.5; // Medium dots ~1.5-2.0px
      }

      const initialColor = pickPaletteColor(initialPalette);
      const dotBaseAlpha = (0.28 + Math.random() * 0.42) * effectiveBaseOpacity;

      particles.push({
        relX,
        relY,
        originX,
        originY,
        x: originX,
        y: originY,
        vx: 0,
        vy: 0,
        baseRadius,
        currentRadius: baseRadius,
        rgb: [...initialColor],
        targetRgb: [...initialColor],
        baseAlpha: dotBaseAlpha,
        targetBaseAlpha: dotBaseAlpha,
        currentAlpha: dotBaseAlpha,
        phaseX: Math.random() * Math.PI * 2,
        phaseY: Math.random() * Math.PI * 2,
        speed: ambientSpeed * (0.6 + Math.random() * 0.8),
        ambientRadius: 4 + Math.random() * 8,
        isAccent,
      });
    }

    particlesRef.current = particles;

    // Handle Resize
    const resizeCanvas = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
      ctx.scale(dpr, dpr);

      // Recompute origins from relative positions so pattern scales seamlessly
      for (const p of particlesRef.current) {
        p.originX = p.relX * width;
        p.originY = p.relY * height;
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas, { passive: true });

    // Pointer Event Listeners
    const onPointerMove = (e: MouseEvent | TouchEvent) => {
      let clientX = 0;
      let clientY = 0;

      if ('touches' in e && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else if ('clientX' in e) {
        clientX = (e as MouseEvent).clientX;
        clientY = (e as MouseEvent).clientY;
      } else {
        return;
      }

      const ptr = pointerRef.current;
      if (ptr.prevX !== -9999) {
        ptr.vx = (clientX - ptr.prevX) * 0.4;
        ptr.vy = (clientY - ptr.prevY) * 0.4;
      }
      ptr.prevX = clientX;
      ptr.prevY = clientY;
      ptr.x = clientX;
      ptr.y = clientY;
      ptr.active = true;
      ptr.lastActiveTime = performance.now();
    };

    const onPointerLeave = () => {
      const ptr = pointerRef.current;
      ptr.active = false;
      ptr.x = -9999;
      ptr.y = -9999;
      ptr.vx = 0;
      ptr.vy = 0;
    };

    window.addEventListener('mousemove', onPointerMove, { passive: true });
    window.addEventListener('touchmove', onPointerMove, { passive: true });
    document.addEventListener('mouseleave', onPointerLeave);
    window.addEventListener('blur', onPointerLeave);

    // If user prefers reduced motion, render static dots once and exit
    if (prefersReducedMotion) {
      ctx.clearRect(0, 0, width, height);
      for (const p of particlesRef.current) {
        ctx.fillStyle = `rgba(${p.rgb[0]}, ${p.rgb[1]}, ${p.rgb[2]}, ${p.baseAlpha * 0.7})`;
        ctx.beginPath();
        ctx.arc(p.originX, p.originY, p.baseRadius, 0, Math.PI * 2);
        ctx.fill();
      }
      return () => {
        window.removeEventListener('resize', resizeCanvas);
        window.removeEventListener('mousemove', onPointerMove);
        window.removeEventListener('touchmove', onPointerMove);
        document.removeEventListener('mouseleave', onPointerLeave);
        window.removeEventListener('blur', onPointerLeave);
      };
    }

    // Animation loop
    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const deltaTime = Math.min((currentTime - lastTime) / 1000, 0.1);
      lastTime = currentTime;

      const ptr = pointerRef.current;

      // Inactivity check: smoothly fade out cursor influence if mouse has stopped moving for > 2.5s
      if (ptr.active && currentTime - ptr.lastActiveTime > 2500) {
        ptr.vx *= 0.85;
        ptr.vy *= 0.85;
      } else {
        ptr.vx *= 0.92;
        ptr.vy *= 0.92;
      }

      ctx.clearRect(0, 0, width, height);

      const radiusSq = interactionRadius * interactionRadius;
      const pts = particlesRef.current;
      const currentIsLight = !currentThemeRef.current?.isDark;

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];

        // 1. Smooth Color & Base Alpha Morphing (Lerp 5% per frame for smooth ~300ms transition)
        p.rgb[0] += (p.targetRgb[0] - p.rgb[0]) * 0.05;
        p.rgb[1] += (p.targetRgb[1] - p.rgb[1]) * 0.05;
        p.rgb[2] += (p.targetRgb[2] - p.rgb[2]) * 0.05;
        p.baseAlpha += (p.targetBaseAlpha - p.baseAlpha) * 0.05;

        // 2. Calculate ambient drift target
        const ambX = p.originX + Math.sin(p.phaseX + currentTime * p.speed) * p.ambientRadius;
        const ambY = p.originY + Math.cos(p.phaseY + currentTime * p.speed) * p.ambientRadius;

        // 3. Cursor repulsion & Google Antigravity wave ripple force
        let targetAlpha = p.baseAlpha;
        let targetRadius = p.baseRadius;

        if (ptr.active) {
          const dx = p.x - ptr.x;
          const dy = p.y - ptr.y;
          const distSq = dx * dx + dy * dy;

          if (distSq < radiusSq && distSq > 0.0001) {
            const dist = Math.sqrt(distSq);
            // Smooth bell curve force: strongest near center, tapering gently
            const factor = 1 - dist / interactionRadius;
            const force = factor * factor * (3 - 2 * factor); // Cubic smoothstep

            const nx = dx / dist;
            const ny = dy / dist;

            // Concentric wave ripple inspired by Google Antigravity
            const wave = Math.sin(dist * 0.045 - currentTime * 0.005) * 0.28;

            // Fluid swirl wake displacement based on cursor velocity
            const swirlX = -ny * 0.2;
            const swirlY = nx * 0.2;

            const pushMag = (force + wave * factor) * cursorForce * 320;
            const wakeMag = force * 0.45;

            p.vx += (nx + swirlX) * pushMag * deltaTime + ptr.vx * wakeMag;
            p.vy += (ny + swirlY) * pushMag * deltaTime + ptr.vy * wakeMag;

            // Subtle glow/brightness amplification near cursor
            targetAlpha = Math.min(0.95, p.baseAlpha + (1 - p.baseAlpha) * force * 0.75);
            targetRadius = p.baseRadius * (1 + force * 0.45);
          }
        }

        // 4. Smooth spring-back force toward ambient target position
        const springX = (ambX - p.x) * springStrength;
        const springY = (ambY - p.y) * springStrength;

        p.vx = (p.vx + springX) * friction;
        p.vy = (p.vy + springY) * friction;

        p.x += p.vx;
        p.y += p.vy;

        // Smooth transition for alpha & radius
        p.currentAlpha += (targetAlpha - p.currentAlpha) * 0.08;
        p.currentRadius += (targetRadius - p.currentRadius) * 0.08;

        // 5. Render Particle Dot with rounded integer RGB
        const r = Math.round(p.rgb[0]);
        const g = Math.round(p.rgb[1]);
        const b = Math.round(p.rgb[2]);

        // Draw soft ambient halo for accent particles or particles excited by cursor
        if (p.isAccent || p.currentAlpha > 0.65) {
          const haloAlpha = currentIsLight ? p.currentAlpha * 0.12 : p.currentAlpha * 0.20;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${haloAlpha})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.currentRadius * 2.2, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.currentAlpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.currentRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    // Pause animation when tab is inactive/hidden to conserve battery and GPU
    const onVisibilityChange = () => {
      if (document.hidden) {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      } else {
        lastTime = performance.now();
        if (!animationFrameRef.current) {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      }
    };

    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', onPointerMove);
      window.removeEventListener('touchmove', onPointerMove);
      document.removeEventListener('mouseleave', onPointerLeave);
      window.removeEventListener('blur', onPointerLeave);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [
    particleCount,
    mobileParticleCount,
    interactionRadius,
    cursorForce,
    springStrength,
    friction,
    ambientSpeed,
    baseOpacity,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="interactive-particle-canvas"
      aria-hidden="true"
    />
  );
};

export default InteractiveParticleBackground;
