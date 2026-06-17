// viz.jsx — shared helpers + chart components (themed via `theme` prop)
// Exports to window at the end (each Babel script has its own scope).

const BCA_REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function deNum(n, dec = 0) {
  return Number(n).toLocaleString('de-DE', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function useInView(threshold = 0.15) {
  const ref = React.useRef(null);
  const [vis, setVis] = React.useState(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!('IntersectionObserver' in window)) { setVis(true); return; }
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) setVis(true); });
    }, { threshold });
    io.observe(el);
    // Fallback: headless iframes / clipped containers never intersect —
    // never leave content stuck at opacity 0.
    const t = setTimeout(() => setVis(true), 900);
    return () => { io.disconnect(); clearTimeout(t); };
  }, []);
  return [ref, vis];
}

// Count-up number. anim: 'aus' | 'dezent' | 'voll'
function CountUp({ value, decimals = 0, anim = 'dezent', suffix = '' }) {
  const [ref, vis] = useInView(0.4);
  const dur = (BCA_REDUCED || anim === 'aus') ? 0 : anim === 'voll' ? 1200 : 700;
  const [v, setV] = React.useState(dur ? 0 : value);
  React.useEffect(() => {
    if (!dur) { setV(value); return; }
    if (!vis) return;
    let raf;
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      setV(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    // rAF is paused in hidden iframes — always settle on the final value.
    const settle = setTimeout(() => setV(value), dur + 150);
    return () => { cancelAnimationFrame(raf); clearTimeout(settle); };
  }, [vis, value, dur]);
  return <span ref={ref}>{deNum(dur ? v : value, decimals)}{suffix}</span>;
}

// Staggered entrance wrapper
function Reveal({ i = 0, anim = 'dezent', children, style }) {
  const [ref, vis] = useInView(0.08);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  const dy = anim === 'voll' ? 20 : 10;
  const step = anim === 'voll' ? 95 : 60;
  return (
    <div ref={ref} style={{
      opacity: on ? 1 : 0,
      transform: on ? 'none' : 'translateY(' + dy + 'px)',
      transition: 'opacity .5s ease ' + (i * step) + 'ms, transform .55s cubic-bezier(.22,.8,.3,1) ' + (i * step) + 'ms',
      ...style
    }}>{children}</div>
  );
}

// ── Heat grid from gaussian spots ──────────────────────────────
function heatGrid(spots, cols, rows) {
  const g = [];
  let max = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) / cols, y = (r + 0.5) / rows;
      let v = 0;
      spots.forEach((s) => {
        const dx = x - s.x, dy = y - s.y;
        v += s.w * Math.exp(-(dx * dx + dy * dy) / (2 * s.s * s.s));
      });
      g.push({ c, r, v });
      if (v > max) max = v;
    }
  }
  g.forEach((cell) => { cell.v = cell.v / max; });
  return g;
}

// ── Court heatmap (half court, top-down, net at top) ───────────
// style: 'blobs' | 'raster' | 'punkte';  1 m = 40 px → 244 × 268
function CourtHeatmap({ spots, style = 'blobs', dark = false, width = 232, anim = 'dezent' }) {
  const [ref, vis] = useInView(0.25);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  const uid = React.useId().replace(/[^a-zA-Z0-9]/g, '');
  const W = 244, H = 268;
  const cols = 12, rows = 13;
  const grid = React.useMemo(() => heatGrid(spots, cols, rows), [spots]);
  const cw = W / cols, ch = H / rows;
  const line = dark ? 'rgba(255,255,255,.26)' : '#DDD5CC';
  const courtBg = dark ? '#16161A' : (style === 'punkte' ? 'transparent' : '#FFFFFF');
  const orange = '#FA8832', red = '#C0392B', ink = '#0A0A0B';

  const lines = (
    <g stroke={line} strokeWidth="2" fill="none">
      <rect x="1" y="1" width={W - 2} height={H - 2} />
      <line x1="18.4" y1="1" x2="18.4" y2={H - 1} />
      <line x1={W - 18.4} y1="1" x2={W - 18.4} y2={H - 1} />
      <line x1="1" y1="79" x2={W - 1} y2="79" />
      <line x1="1" y1={H - 30} x2={W - 1} y2={H - 30} />
      <line x1={W / 2} y1="79" x2={W / 2} y2={H - 1} />
      <line x1="0" y1="3" x2={W} y2="3" stroke={dark ? 'rgba(255,255,255,.6)' : '#B9AFA4'} strokeWidth="4" />
    </g>
  );

  let heat = null;
  if (style === 'blobs') {
    heat = (
      <g filter={'url(#blur' + uid + ')'} style={{ opacity: on ? 1 : 0, transition: 'opacity .9s ease .15s' }}>
        {spots.map((s, i) => (
          <circle key={i} cx={s.x * W} cy={s.y * H} r={s.s * 430 * (0.55 + 0.45 * s.w)}
            fill={'url(#rg' + uid + ')'} opacity={0.4 + 0.55 * s.w} />
        ))}
      </g>
    );
  } else if (style === 'raster') {
    heat = (
      <g>
        {grid.map((cell) => cell.v > 0.05 ? (
          <rect key={cell.r + '-' + cell.c} x={cell.c * cw + 1.6} y={cell.r * ch + 1.6}
            width={cw - 3.2} height={ch - 3.2} rx="3"
            fill={cell.v > 0.85 ? red : orange}
            style={{
              opacity: on ? Math.min(0.95, 0.07 + cell.v * 0.88) : 0,
              transition: 'opacity .45s ease ' + ((cell.c + cell.r) * 22) + 'ms'
            }} />
        ) : null)}
      </g>
    );
  } else {
    heat = (
      <g>
        {grid.map((cell) => cell.v > 0.06 ? (
          <circle key={cell.r + '-' + cell.c} cx={(cell.c + 0.5) * cw} cy={(cell.r + 0.5) * ch}
            r={1.4 + cell.v * 6.6}
            fill={cell.v > 0.72 ? orange : (dark ? '#FFFFFF' : ink)}
            style={{
              opacity: on ? 0.14 + cell.v * 0.78 : 0,
              transition: 'opacity .45s ease ' + ((cell.c + cell.r) * 22) + 'ms'
            }} />
        ) : null)}
      </g>
    );
  }

  return (
    <div ref={ref} style={{ lineHeight: 0 }}>
      <svg width={width} height={width * H / W} viewBox={'0 0 ' + W + ' ' + H} style={{ display: 'block' }}>
        <defs>
          <radialGradient id={'rg' + uid}>
            <stop offset="0%" stopColor={red} stopOpacity="0.85" />
            <stop offset="38%" stopColor={orange} stopOpacity="0.7" />
            <stop offset="100%" stopColor={orange} stopOpacity="0" />
          </radialGradient>
          <filter id={'blur' + uid} x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="9" />
          </filter>
          <clipPath id={'clip' + uid}><rect x="0" y="0" width={W} height={H} /></clipPath>
        </defs>
        <rect x="0" y="0" width={W} height={H} fill={courtBg} rx={dark ? 6 : 0} />
        <g clipPath={'url(#clip' + uid + ')'}>{heat}</g>
        {lines}
      </svg>
    </div>
  );
}

// Small legend under heatmaps
function HeatLegend({ theme }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
      <span style={{ fontSize: 11, color: theme.sub }}>wenig</span>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'linear-gradient(90deg, rgba(250,136,50,.08), #FA8832 60%, #C0392B)' }}></div>
      <span style={{ fontSize: 11, color: theme.sub }}>viel</span>
    </div>
  );
}

// ── Zones: front/mid/rear bands + left/right split ─────────────
function ZoneBlock({ zones, theme, anim = 'dezent' }) {
  const [ref, vis] = useInView(0.3);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  const bands = [
    { label: 'Vorderfeld', v: zones.front },
    { label: 'Mittelfeld', v: zones.mid },
    { label: 'Hinterfeld', v: zones.rear }
  ];
  const maxV = Math.max(zones.front, zones.mid, zones.rear);
  const mono = theme.key === 'D';
  const barBg = theme.key === 'B' || theme.key === 'D' ? (theme.key === 'D' ? 'rgba(10,10,11,.08)' : '#F1EFEC') : 'rgba(250,136,50,.12)';

  return (
    <div ref={ref}>
      <div style={{ display: 'grid', gap: 10 }}>
        {bands.map((b, i) => (
          <div key={b.label} style={{ display: 'grid', gridTemplateColumns: '86px 1fr 44px', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 12.5, color: theme.sub, fontFamily: mono ? 'var(--font-mono)' : 'inherit' }}>{b.label}</span>
            <div style={{ height: mono ? 8 : 12, borderRadius: 6, background: barBg, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 6,
                width: (b.v / maxV * 100) + '%',
                background: b.v === maxV ? theme.accent : (theme.key === 'B' ? '#0A0A0B' : theme.accentSoft),
                transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'left',
                transition: 'transform .7s cubic-bezier(.22,.8,.3,1) ' + (i * 110) + 'ms'
              }}></div>
            </div>
            <strong style={{
              fontSize: 14, textAlign: 'right', fontFamily: mono ? 'var(--font-mono)' : 'var(--font-heading)',
              fontVariantNumeric: 'tabular-nums'
            }}>{b.v}&thinsp;%</strong>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: theme.sub, marginBottom: 6 }}>
          <span>Links {zones.left}&thinsp;%</span><span>Rechts {zones.right}&thinsp;%</span>
        </div>
        <div style={{ display: 'flex', height: mono ? 8 : 12, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          <div style={{
            width: zones.left + '%', background: theme.key === 'B' ? '#0A0A0B' : theme.accent, borderRadius: '6px 0 0 6px',
            transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'left', transition: 'transform .7s cubic-bezier(.22,.8,.3,1) .25s'
          }}></div>
          <div style={{
            width: zones.right + '%', background: barBg, borderRadius: '0 6px 6px 0',
            transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'right', transition: 'transform .7s cubic-bezier(.22,.8,.3,1) .25s'
          }}></div>
        </div>
      </div>
    </div>
  );
}

// ── Activity timeline: distance per 5-minute bucket ────────────
function TimelineChart({ points, theme, anim = 'dezent' }) {
  const [ref, vis] = useInView(0.3);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  const W = 330, H = 120, pad = 2, base = 96;
  const max = Math.max(...points);
  const maxI = points.indexOf(max);
  const n = points.length;

  if (theme.timelineStyle === 'line') {
    const pts = points.map((v, i) => {
      const x = pad + (i / (n - 1)) * (W - pad * 2);
      const y = base - (v / max) * 78;
      return [x, y];
    });
    const path = 'M' + pts.map((p) => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
    return (
      <div ref={ref}>
        <svg width="100%" viewBox={'0 0 ' + W + ' ' + H} style={{ display: 'block' }}>
          <line x1="0" y1={base} x2={W} y2={base} stroke="rgba(10,10,11,.18)" strokeWidth="1" />
          <path d={path} fill="none" stroke="#0A0A0B" strokeWidth="1.5" pathLength="1"
            strokeDasharray="1" strokeDashoffset={on ? 0 : 1}
            style={{ transition: 'stroke-dashoffset 1.1s ease .1s' }} />
          {pts.map((p, i) => (
            <circle key={i} cx={p[0]} cy={p[1]} r={i === maxI ? 4 : 2.4}
              fill={i === maxI ? theme.accent : '#0A0A0B'}
              style={{ opacity: on ? 1 : 0, transition: 'opacity .3s ease ' + (200 + i * 70) + 'ms' }} />
          ))}
          {points.map((v, i) => i % 2 === 0 ? (
            <text key={'t' + i} x={pad + (i / (n - 1)) * (W - pad * 2)} y={H - 8} textAnchor="middle"
              fontSize="9.5" fill={theme.sub} fontFamily="var(--font-mono)">{i * 5}&prime;</text>
          ) : null)}
        </svg>
      </div>
    );
  }

  const bw = W / n;
  return (
    <div ref={ref}>
      <svg width="100%" viewBox={'0 0 ' + W + ' ' + H} style={{ display: 'block' }}>
        {points.map((v, i) => {
          const h = (v / max) * 82;
          const hot = i === maxI;
          return (
            <g key={i}>
              <rect x={i * bw + 3} y={base - h} width={bw - 6} height={h} rx={theme.key === 'B' ? 1 : 4}
                fill={hot ? theme.accent : (theme.key === 'B' ? '#0A0A0B' : theme.accentSoft)}
                style={{
                  transform: on ? 'scaleY(1)' : 'scaleY(0)', transformOrigin: 'bottom', transformBox: 'fill-box',
                  transition: 'transform .55s cubic-bezier(.22,.8,.3,1) ' + (i * 55) + 'ms'
                }} />
              {hot ? (
                <text x={i * bw + bw / 2} y={base - h - 7} textAnchor="middle" fontSize="10.5" fontWeight="700"
                  fill={theme.ink} fontFamily="var(--font-heading)"
                  style={{ opacity: on ? 1 : 0, transition: 'opacity .4s ease ' + (300 + i * 55) + 'ms' }}>{v}&thinsp;m</text>
              ) : null}
              {i % 2 === 0 ? (
                <text x={i * bw + bw / 2} y={H - 8} textAnchor="middle" fontSize="9.5" fill={theme.sub}>{i * 5}&prime;</text>
              ) : null}
            </g>
          );
        })}
        <line x1="0" y1={base + 1} x2={W} y2={base + 1} stroke={theme.key === 'C' ? 'rgba(10,10,11,.14)' : 'rgba(10,10,11,.12)'} strokeWidth="1" />
      </svg>
    </div>
  );
}

// ── Shot mix: horizontal bars ──────────────────────────────────
function ShotMix({ shots, theme, anim = 'dezent', compact }) {
  const [ref, vis] = useInView(0.2);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  const total = shots.reduce((a, s) => a + s[1], 0);
  const sorted = [...shots].sort((a, b) => b[1] - a[1]);
  const max = sorted[0][1];
  const mono = theme.key === 'D';
  const barBg = theme.key === 'D' ? 'rgba(10,10,11,.08)' : theme.key === 'B' ? '#F1EFEC' : 'rgba(250,136,50,.12)';
  return (
    <div ref={ref} style={{ display: 'grid', gap: compact ? 6.5 : 9 }}>
      {sorted.map((s, i) => {
        const pct = Math.round(s[1] / total * 100);
        const top = i === 0;
        return (
          <div key={s[0]} style={{ display: 'grid', gridTemplateColumns: compact ? '70px 1fr 54px' : '76px 1fr 58px', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: compact ? 11.5 : 12.5, color: top ? theme.ink : theme.sub, fontWeight: top ? 600 : 400, fontFamily: mono ? 'var(--font-mono)' : 'inherit' }}>{s[0]}</span>
            <div style={{ height: compact ? 8 : (mono ? 7 : 10), borderRadius: 5, background: barBg, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 5, width: (s[1] / max * 100) + '%',
                background: top ? theme.accent : (theme.key === 'B' ? '#0A0A0B' : theme.accentSoft),
                transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'left',
                transition: 'transform .6s cubic-bezier(.22,.8,.3,1) ' + (i * 70) + 'ms'
              }}></div>
            </div>
            <span style={{
              fontSize: compact ? 12 : 13, textAlign: 'right', fontVariantNumeric: 'tabular-nums',
              fontFamily: mono ? 'var(--font-mono)' : 'var(--font-heading)', fontWeight: 700
            }}>{s[1]}<span style={{ color: theme.sub, fontWeight: 400, fontSize: compact ? 10.5 : 11 }}> · {pct}%</span></span>
          </div>
        );
      })}
    </div>
  );
}

// ── Doubles formation split ────────────────────────────────────
function FormationBlock({ pairs, theme, anim = 'dezent' }) {
  const [ref, vis] = useInView(0.3);
  const on = BCA_REDUCED || anim === 'aus' ? true : vis;
  return (
    <div ref={ref} style={{ display: 'grid', gap: 18 }}>
      {pairs.map((pair, pi) => (
        <div key={pair.side}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 7 }}>
            <strong style={{ fontSize: 13.5, fontFamily: 'var(--font-heading)' }}>{pair.label}</strong>
            <span style={{ fontSize: 12, color: theme.sub }}>{pair.rot} Rotationen</span>
          </div>
          <div style={{ display: 'flex', height: 14, borderRadius: 7, overflow: 'hidden', gap: 2 }}>
            <div style={{
              width: pair.fb + '%', background: theme.accent, borderRadius: '7px 0 0 7px',
              transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'left',
              transition: 'transform .7s cubic-bezier(.22,.8,.3,1) ' + (pi * 140) + 'ms'
            }}></div>
            <div style={{
              width: pair.ss + '%', background: theme.key === 'B' ? '#0A0A0B' : 'rgba(10,10,11,.16)', borderRadius: '0 7px 7px 0',
              transform: on ? 'scaleX(1)' : 'scaleX(0)', transformOrigin: 'right',
              transition: 'transform .7s cubic-bezier(.22,.8,.3,1) ' + (pi * 140 + 80) + 'ms'
            }}></div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: theme.sub, marginTop: 5 }}>
            <span>Vorne/Hinten {pair.fb}&thinsp;%</span><span>Nebeneinander {pair.ss}&thinsp;%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tiny inline icons (stroke, 2px, 24-grid) ───────────────────
function Icon({ d, size = 18, color = 'currentColor', fill = 'none' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color}
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>
      {d}
    </svg>
  );
}
const ICONS = {
  route: <g><circle cx="5" cy="19" r="2"></circle><circle cx="19" cy="5" r="2"></circle><path d="M7 19h7a4 4 0 0 0 0-8H10a4 4 0 0 1 0-8h7"></path></g>,
  gauge: <g><path d="M12 15l3.5-3.5"></path><path d="M20.3 14a8.5 8.5 0 1 0-16.6 0"></path></g>,
  bolt: <g><path d="M13 2 4.5 14H11l-1 8 8.5-12H12l1-8z"></path></g>,
  clock: <g><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path></g>,
  chevR: <path d="M9 6l6 6-6 6"></path>,
  arrowL: <path d="M19 12H5m6-7-7 7 7 7"></path>,
  replay: <g><path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 4v5h5"></path></g>
};

Object.assign(window, {
  deNum, useInView, CountUp, Reveal, heatGrid, CourtHeatmap, HeatLegend,
  ZoneBlock, TimelineChart, ShotMix, FormationBlock, Icon, ICONS, BCA_REDUCED
});
