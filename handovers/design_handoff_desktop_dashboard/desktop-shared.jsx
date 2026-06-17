// desktop-shared.jsx — Desktop-Bausteine für Richtung A (Vereinsfarben)
// Depends on: viz.jsx (charts, ICONS, CountUp), screens.jsx (THEMES, ResultChip),
// video.jsx (VideoOverlay, VidToggleChip, vidFmtTime), mock-data.js

const DK_HAIR = '1px solid hsl(214.3 31.8% 91.4%)';
const DK_SUB = 'hsl(215.4 16.3% 46.9%)';
const DK_INK = '#0A0A0B';
const DK_ORANGE = '#FA8832';
const DK_ORANGE_DARK = '#C66C27';

const DK_ICONS = {
  chevD: <path d="M6 9l6 6 6-6"></path>
};

// ── White card ─────────────────────────────────────────────────
function DeskCard({ title, sub, children, foot, style, dense }) {
  return (
    <section style={Object.assign({
      background: '#FFFFFF', border: DK_HAIR, borderRadius: 14,
      boxShadow: 'var(--shadow-sm)', padding: dense ? '14px 16px 16px' : '18px 20px 20px',
      boxSizing: 'border-box'
    }, style)}>
      {title ? (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 10, marginBottom: dense ? 11 : 14 }}>
          <h3 style={{ margin: 0, fontSize: dense ? 14 : 15.5, fontFamily: 'var(--font-heading)', fontWeight: 700, whiteSpace: 'nowrap' }}>{title}</h3>
          {sub ? <span style={{ fontSize: 11.5, color: DK_SUB, textAlign: 'right' }}>{sub}</span> : null}
        </div>
      ) : null}
      {children}
      {foot ? <p style={{ margin: '12px 0 0', fontSize: 11.5, color: DK_SUB }}>{foot}</p> : null}
    </section>
  );
}

// ── Kennzahlen ─────────────────────────────────────────────────
function dkItems(stats) {
  return [
    { label: 'Distanz', value: stats.distance_m / 1000, dec: 2, unit: 'km', icon: ICONS.route },
    { label: 'Sprints', value: stats.sprints, dec: 0, unit: '', icon: ICONS.bolt },
    { label: 'Ø Tempo', value: stats.avg_kmh, dec: 1, unit: 'km/h', icon: ICONS.gauge },
    { label: 'Top-Tempo', value: stats.top_kmh, dec: 1, unit: 'km/h', icon: ICONS.clock }
  ];
}

function DeskKpiTile({ it, anim }) {
  return (
    <DeskCard dense style={{ height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
        <span style={{ width: 30, height: 30, borderRadius: 10, background: 'rgba(250,136,50,.13)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: DK_ORANGE_DARK, flex: 'none' }}>
          <Icon d={it.icon} size={16} />
        </span>
        <span style={{ fontSize: 12.5, color: DK_SUB, fontWeight: 500, whiteSpace: 'nowrap' }}>{it.label}</span>
      </div>
      <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 30, marginTop: 10, letterSpacing: '-0.01em' }}>
        <CountUp value={it.value} decimals={it.dec} anim={anim} />
        <span style={{ fontSize: 13.5, fontWeight: 600, color: DK_SUB, marginLeft: 5 }}>{it.unit}</span>
      </div>
    </DeskCard>
  );
}

function DeskKpiBare({ it, anim }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <span style={{ width: 24, height: 24, borderRadius: 8, background: 'rgba(250,136,50,.13)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: DK_ORANGE_DARK, flex: 'none' }}>
          <Icon d={it.icon} size={13} />
        </span>
        <span style={{ fontSize: 11.5, color: DK_SUB, whiteSpace: 'nowrap' }}>{it.label}</span>
      </div>
      <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 22, marginTop: 6 }}>
        <CountUp value={it.value} decimals={it.dec} anim={anim} />
        <span style={{ fontSize: 12, fontWeight: 600, color: DK_SUB, marginLeft: 4 }}>{it.unit}</span>
      </div>
    </div>
  );
}

// ── Ballwechsel-Zusammenfassung ────────────────────────────────
function DeskRallies({ rallies, anim, horizontal }) {
  const nums = (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, textAlign: 'center' }}>
      {[[rallies.count, 0, 'gesamt'], [rallies.avgShots, 1, 'Ø Schläge'], [rallies.longShots, 0, 'längster']].map((it) => (
        <div key={it[2]}>
          <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 26 }}>
            <CountUp value={it[0]} decimals={it[1]} anim={anim} />
          </div>
          <div style={{ fontSize: 11.5, color: DK_SUB, marginTop: 2 }}>{it[2]}</div>
        </div>
      ))}
    </div>
  );
  const note = (
    <p style={{ margin: horizontal ? 0 : '12px 0 0', fontSize: 12.5, color: DK_SUB, textAlign: horizontal ? 'left' : 'center', lineHeight: 1.55 }}>
      Längster Ballwechsel: {rallies.longShots} Schläge in {rallies.longSec}&thinsp;s. {rallies.endLine}
    </p>
  );
  if (horizontal) {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 28, alignItems: 'center' }}>
        {nums}{note}
      </div>
    );
  }
  return <div>{nums}{note}</div>;
}

// ── Endstand-Karte (hell) ──────────────────────────────────────
function DeskScoreboard({ report, style, dense, extra }) {
  const m = report.meta;
  const sideName = (side) => {
    if (report.pairs) return report.pairs.find((x) => x.side === side).label;
    const pl = report.players.find((x) => x.side === side);
    return pl.isViewer ? pl.name + ' (du)' : pl.name;
  };
  return (
    <div style={Object.assign({
      background: '#FFFFFF', border: DK_HAIR, borderRadius: 16,
      boxShadow: dense ? 'var(--shadow-sm)' : 'var(--shadow-md)',
      padding: dense ? '14px 16px 16px' : '20px 22px', boxSizing: 'border-box'
    }, style)}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
        <span className="eyebrow" style={{ fontSize: 11 }}>Endstand</span>
        <span style={{ fontSize: 12, color: DK_SUB, whiteSpace: 'nowrap' }}>{m.discipline} · Feld {m.court}</span>
      </div>
      <div style={{ marginTop: dense ? 12 : 16, display: 'grid', gap: 10 }}>
        {['A', 'B'].map((side) => (
          <div key={side} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{
              flex: 1, fontFamily: 'var(--font-heading)', fontWeight: side === 'A' ? 700 : 500,
              fontSize: dense ? 13.5 : 15.5, color: side === 'A' ? DK_INK : DK_SUB,
              minWidth: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'
            }}>{sideName(side)}</span>
            <span style={{ display: 'flex', gap: 6, flex: 'none' }}>
              {m.games.map((g, gi) => {
                const v = side === 'A' ? g[0] : g[1];
                const won = side === 'A' ? g[0] > g[1] : g[1] > g[0];
                return (
                  <span key={gi} style={{
                    width: dense ? 28 : 32, height: dense ? 25 : 28, borderRadius: 7,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: dense ? 13 : 14,
                    fontVariantNumeric: 'tabular-nums',
                    background: won ? DK_ORANGE : 'rgba(10,10,11,.06)',
                    color: won ? '#1A0500' : DK_SUB
                  }}>{v}</span>
                );
              })}
            </span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: dense ? 12 : 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <ResultChip result={m.result} line={m.resultLine} theme={THEMES.A} small={dense} />
        <span style={{ fontSize: dense ? 11.5 : 12, color: DK_SUB, textAlign: 'right' }}>{m.durationMin} Min Spielzeit · {m.restMin} Min Pause</span>
      </div>
      {extra ? (
        <div style={{ marginTop: 11, paddingTop: 10, borderTop: DK_HAIR, fontSize: 11.5, color: DK_SUB, lineHeight: 1.5 }}>{extra}</div>
      ) : null}
    </div>
  );
}

// ── Video-Player (Hook + Bausteine) ────────────────────────────
function useDeskVideo(posId) {
  const vref = React.useRef(null);
  const [playing, setPlaying] = React.useState(false);
  const [time, setTime] = React.useState(0);
  const [dur, setDur] = React.useState(0);
  const posKey = 'bca-vt-' + posId;
  const [ov, setOv] = React.useState(() => {
    try {
      const s = JSON.parse(localStorage.getItem('bca-ov-v1'));
      if (s && typeof s === 'object') return Object.assign({ tracking: true, wege: true, shuttle: false, zonen: false }, s);
    } catch (e) { /* noop */ }
    return { tracking: true, wege: true, shuttle: false, zonen: false };
  });
  const toggle = (key) => {
    const next = Object.assign({}, ov);
    next[key] = !next[key];
    setOv(next);
    try { localStorage.setItem('bca-ov-v1', JSON.stringify(next)); } catch (e) { /* noop */ }
  };
  React.useEffect(() => {
    if (!playing) return;
    let raf;
    const tick = () => { const v = vref.current; if (v) setTime(v.currentTime); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
  React.useEffect(() => {
    const id = Math.floor(time);
    if (id > 0) { try { localStorage.setItem(posKey, String(id)); } catch (e) { /* noop */ } }
  }, [Math.floor(time)]);
  const onLoaded = () => {
    const v = vref.current;
    if (!v) return;
    setDur(v.duration || 0);
    try {
      const saved = parseFloat(localStorage.getItem(posKey));
      if (saved > 0 && saved < (v.duration || 0) - 1) { v.currentTime = saved; setTime(saved); }
      else { v.currentTime = 1; }
    } catch (e) { v.currentTime = 1; }
  };
  const togglePlay = () => {
    const v = vref.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); } else { v.pause(); setPlaying(false); }
  };
  const seek = (val) => {
    const v = vref.current;
    if (!v || !dur) return;
    v.currentTime = val;
    setTime(val);
  };
  return { vref, playing, setPlaying, time, setTime, dur, ov, toggle, onLoaded, togglePlay, seek };
}

function DeskVideoSurface({ vp, video, players, radius = 12, fill }) {
  if (fill) {
    // Höhenbegrenzte 16:9-Bühne: füllt die verfügbare Höhe, Breite folgt dem Seitenverhältnis
    return (
      <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0A0A0B' }}>
        <div style={{ position: 'relative', height: '100%', maxWidth: '100%', aspectRatio: '16 / 9', lineHeight: 0 }}>
          <video ref={vp.vref} src={video.src} muted playsInline preload="metadata"
            onLoadedMetadata={vp.onLoaded} onEnded={() => vp.setPlaying(false)}
            onTimeUpdate={() => { const v = vp.vref.current; if (v && v.paused) vp.setTime(v.currentTime); }}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}></video>
          <VideoOverlay time={vp.time} players={players} ov={vp.ov} />
        </div>
      </div>
    );
  }
  return (
    <div style={{ position: 'relative', borderRadius: radius, overflow: 'hidden', background: '#0A0A0B', lineHeight: 0 }}>
      <video ref={vp.vref} src={video.src} muted playsInline preload="metadata"
        onLoadedMetadata={vp.onLoaded} onEnded={() => vp.setPlaying(false)}
        onTimeUpdate={() => { const v = vp.vref.current; if (v && v.paused) vp.setTime(v.currentTime); }}
        style={{ width: '100%', display: 'block' }}></video>
      <VideoOverlay time={vp.time} players={players} ov={vp.ov} />
    </div>
  );
}

function DeskVideoControls({ vp, style }) {
  return (
    <div style={Object.assign({ display: 'flex', alignItems: 'center', gap: 10 }, style)}>
      <button type="button" onClick={vp.togglePlay} aria-label={vp.playing ? 'Pause' : 'Abspielen'} style={{
        width: 38, height: 38, borderRadius: 19, border: '1px solid #C66C27', cursor: 'pointer',
        background: '#FA8832', color: '#1A0500', display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 3px 0 #C66C27', flex: 'none', padding: 0
      }}>
        {vp.playing ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="5" height="16" rx="1.2"></rect><rect x="14" y="4" width="5" height="16" rx="1.2"></rect></svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.8v14.4c0 .9 1 1.5 1.8 1L20 13c.8-.5.8-1.6 0-2.1L8.8 3.9C8 3.4 7 4 7 4.8z"></path></svg>
        )}
      </button>
      <input type="range" min="0" max={vp.dur || 0} step="0.1" value={vp.time}
        onChange={(e) => vp.seek(parseFloat(e.target.value))} className="bca-range"
        style={{ flex: 1, accentColor: '#FA8832' }} />
      <span style={{ fontSize: 11.5, color: DK_SUB, fontVariantNumeric: 'tabular-nums', fontFamily: 'var(--font-mono)', flex: 'none' }}>
        {vidFmtTime(vp.time)} / {vidFmtTime(vp.dur)}
      </span>
    </div>
  );
}

function DeskOverlayChips({ vp }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      <VidToggleChip label="Tracking" on={vp.ov.tracking} onClick={() => vp.toggle('tracking')} />
      <VidToggleChip label="Laufwege" on={vp.ov.wege} onClick={() => vp.toggle('wege')} />
      <VidToggleChip label="Shuttle" on={vp.ov.shuttle} onClick={() => vp.toggle('shuttle')} />
      <VidToggleChip label="Zonen" on={vp.ov.zonen} onClick={() => vp.toggle('zonen')} />
    </div>
  );
}

// Komplette Video-Karte (Karte + Chips + Controls)
function DeskVideoCard({ video, players, posId, dense, style }) {
  const vp = useDeskVideo(posId);
  return (
    <DeskCard title="Video-Analyse" sub={video.label} dense={dense} style={style}
      foot="Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage.">
      <DeskVideoSurface vp={vp} video={video} players={players} />
      <div style={{ marginTop: 12 }}><DeskOverlayChips vp={vp} /></div>
      <DeskVideoControls vp={vp} style={{ marginTop: 12 }} />
    </DeskCard>
  );
}

// ── Safari-ähnlicher Fensterrahmen (hell, macOS) ─────────────
function SafariFrame({ url, children, style, barH = 44 }) {
  const dim = 'rgba(10,10,11,.35)';
  const chev = (d, key) => (
    <svg key={key} width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={dim} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>{d}</svg>
  );
  return (
    <div style={Object.assign({
      borderRadius: 12, overflow: 'hidden', border: DK_HAIR,
      background: '#FFFFFF', boxShadow: 'var(--shadow-md)'
    }, style)}>
      <div style={{ height: barH, flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: '0 12px', background: '#F4F1ED', borderBottom: DK_HAIR, boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', gap: 7, flex: 'none' }}>
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#FF5F57', display: 'block' }}></span>
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#FEBC2E', display: 'block' }}></span>
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#28C840', display: 'block' }}></span>
        </div>
        <div style={{ display: 'flex', gap: 8, flex: 'none', marginLeft: 2 }}>
          {chev(<path d="M15 6l-6 6 6 6"></path>, 'b')}
          {chev(<path d="M9 6l6 6-6 6"></path>, 'f')}
        </div>
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, height: 28, borderRadius: 8, background: 'rgba(10,10,11,.055)', padding: '0 12px', maxWidth: 380, width: '100%', boxSizing: 'border-box' }}>
            <svg width="10" height="12" viewBox="0 0 12 14" fill="none" stroke={dim} strokeWidth="1.6" style={{ flex: 'none' }}><rect x="1.5" y="6" width="9" height="6.5" rx="1.5"></rect><path d="M3.5 6V4.5a2.5 2.5 0 0 1 5 0V6"></path></svg>
            <span style={{ fontSize: 12, color: 'rgba(10,10,11,.6)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{url}</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, flex: 'none' }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={dim} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}><path d="M12 3v11"></path><path d="M8 6l4-3 4 3"></path><path d="M6 10H5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1h-1"></path></svg>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={dim} strokeWidth="2.2" strokeLinecap="round" style={{ display: 'block' }}><path d="M12 5v14M5 12h14"></path></svg>
        </div>
      </div>
      {children}
    </div>
  );
}

Object.assign(window, {
  DK_HAIR, DK_SUB, DK_INK, DK_ORANGE, DK_ORANGE_DARK, DK_ICONS,
  DeskCard, dkItems, DeskKpiTile, DeskKpiBare, DeskRallies, DeskScoreboard,
  useDeskVideo, DeskVideoSurface, DeskVideoControls, DeskOverlayChips, DeskVideoCard, SafariFrame
});
