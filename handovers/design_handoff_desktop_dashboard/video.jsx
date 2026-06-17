// video.jsx — Video-Analyse: gameplay clip + toggleable overlays
// (Tracking-Boxen, Laufwege, Shuttle-Flugbahn, Zonen). Mock tracking data,
// court-mapped onto the fixed-camera frame via a perspective quad.
// VideoOverlay is standalone so desktop layouts can reuse it.
// Depends on viz.jsx + screens.jsx (SectionCard).

// Court quad in video-frame fractions (tuned against assets/match-clip.mp4)
const VID_QUAD = { farY: 0.415, nearY: 0.955, farXL: 0.40, farXR: 0.615, nearXL: 0.34, nearXR: 0.72, k: 0.5625 };
const VID_VB_W = 192, VID_VB_H = 108; // svg viewBox units over the video

// v: 0 = far baseline, 1 = near baseline (net at 0.5) ; u: 0 = left, 1 = right
function vidProject(u, v) {
  const q = VID_QUAD;
  const t = (v * q.k) / (1 + (q.k - 1) * v); // perspective depth
  const y = q.farY + t * (q.nearY - q.farY);
  const xl = q.farXL + t * (q.nearXL - q.farXL);
  const xr = q.farXR + t * (q.nearXR - q.farXR);
  return { x: (xl + u * (xr - xl)) * VID_VB_W, y: y * VID_VB_H, t };
}

// Smooth keyframe track: keys [[time01, u, v], ...] over a loop
function vidTrackAt(keys, p) {
  if (p <= keys[0][0]) return { u: keys[0][1], v: keys[0][2] };
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i], b = keys[i + 1];
    if (p >= a[0] && p <= b[0]) {
      let s = (p - a[0]) / (b[0] - a[0]);
      s = s * s * (3 - 2 * s); // smoothstep
      return { u: a[1] + (b[1] - a[1]) * s, v: a[2] + (b[2] - a[2]) * s };
    }
  }
  const last = keys[keys.length - 1];
  return { u: last[1], v: last[2] };
}

const VID_LOOP = 16; // seconds per movement loop
const VID_TRACK_NEAR = [
  [0, 0.5, 0.80], [0.12, 0.32, 0.66], [0.22, 0.60, 0.62], [0.34, 0.52, 0.90],
  [0.46, 0.70, 0.70], [0.58, 0.44, 0.62], [0.70, 0.30, 0.82], [0.82, 0.62, 0.84], [1, 0.5, 0.80]
];
const VID_TRACK_FAR = [
  [0, 0.5, 0.22], [0.10, 0.66, 0.36], [0.20, 0.40, 0.14], [0.32, 0.56, 0.30],
  [0.44, 0.34, 0.40], [0.56, 0.60, 0.12], [0.68, 0.46, 0.34], [0.80, 0.64, 0.20], [0.90, 0.40, 0.30], [1, 0.5, 0.22]
];

function vidFmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return m + ':' + (ss < 10 ? '0' : '') + ss;
}

// Shuttle: alternating hits every VID_LOOP/10, near→far→near…
function vidShuttleAt(p) {
  const seg = Math.floor(p * 10) % 10;
  const segP = (p * 10) % 1;
  const tA = (seg / 10), tB = ((seg + 1) / 10);
  const fromNear = seg % 2 === 0;
  const a = vidTrackAt(fromNear ? VID_TRACK_NEAR : VID_TRACK_FAR, tA);
  const b = vidTrackAt(fromNear ? VID_TRACK_FAR : VID_TRACK_NEAR, tB > 1 ? tB - 1 : tB);
  const u = a.u + (b.u - a.u) * segP;
  const v = a.v + (b.v - a.v) * segP;
  return { u, v, segP, from: { u: a.u, v: a.v } };
}

function VidToggleChip({ label, on, onClick }) {
  return (
    <button type="button" onClick={onClick} className="bca-tab" style={{
      padding: '5px 12px', borderRadius: 99, fontSize: 12, fontWeight: 600, cursor: 'pointer',
      fontFamily: 'var(--font-body)', transition: 'background .15s ease, color .15s ease, border-color .15s ease',
      background: on ? '#FA8832' : '#FFFFFF',
      color: on ? '#1A0500' : 'hsl(215.4 16.3% 46.9%)',
      border: '1px solid ' + (on ? '#C66C27' : 'hsl(214.3 31.8% 91.4%)'),
      boxShadow: on ? '0 2px 0 #C66C27' : 'none'
    }}>{label}</button>
  );
}

// ── Overlay SVG (absolute-positioned over a 16:9 video) ────────
function VideoOverlay({ time, players, ov }) {
  const p = (time % VID_LOOP) / VID_LOOP;
  const near = players[0] || { first: 'Du' };
  const far = players[1] || { first: 'Gegner' };
  const nearPos = vidTrackAt(VID_TRACK_NEAR, p);
  const farPos = vidTrackAt(VID_TRACK_FAR, p);
  const ORANGE = '#FA8832', BLUE = '#1758A6';

  const playerBox = (pos, name, color) => {
    const pt = vidProject(pos.u, pos.v);
    const w = 4.5 + pt.t * 4.5, h = 8 + pt.t * 7;
    const chipW = Math.max(w, name.length * 2.0 + 2.4);
    return (
      <g>
        <rect x={pt.x - w / 2} y={pt.y - h} width={w} height={h} rx="1"
          fill={color + '14'} stroke={color} strokeWidth="0.45" />
        <rect x={pt.x - w / 2} y={pt.y - h - 4.4} width={chipW} height="3.6" rx="1" fill={color} />
        <text x={pt.x - w / 2 + 1.2} y={pt.y - h - 1.7} fontSize="2.7" fontWeight="700" fill="#FFFFFF"
          fontFamily="var(--font-body)">{name}</text>
        <ellipse cx={pt.x} cy={pt.y} rx={w * 0.62} ry="1.1" fill={color} opacity="0.35" />
      </g>
    );
  };

  const trail = (keys, color) => {
    const pts = [];
    for (let dt = 5; dt >= 0; dt -= 0.25) {
      let tp = ((time - dt) % VID_LOOP) / VID_LOOP;
      if (tp < 0) tp += 1;
      const pos = vidTrackAt(keys, tp);
      const pr = vidProject(pos.u, pos.v);
      pts.push(pr.x.toFixed(1) + ',' + pr.y.toFixed(1));
    }
    return (
      <g fill="none" strokeLinecap="round" strokeLinejoin="round">
        <polyline points={pts.slice(0, 11).join(' ')} stroke={color} strokeWidth="0.55" opacity="0.25" />
        <polyline points={pts.slice(10).join(' ')} stroke={color} strokeWidth="0.8" opacity="0.65" />
      </g>
    );
  };

  const zones = () => {
    const bands = [
      { v0: 0.5, v1: 0.65, pct: 31, label: 'Vorne' },
      { v0: 0.65, v1: 0.82, pct: 44, label: 'Mitte' },
      { v0: 0.82, v1: 1.0, pct: 25, label: 'Hinten' }
    ];
    return (
      <g>
        {bands.map((b) => {
          const tl = vidProject(0, b.v0), tr = vidProject(1, b.v0);
          const bl = vidProject(0, b.v1), br = vidProject(1, b.v1);
          return (
            <g key={b.label}>
              <polygon points={tl.x + ',' + tl.y + ' ' + tr.x + ',' + tr.y + ' ' + br.x + ',' + br.y + ' ' + bl.x + ',' + bl.y}
                fill={ORANGE} opacity={0.10 + b.pct / 100 * 0.5} stroke="#FFFFFF" strokeWidth="0.25" strokeOpacity="0.5" />
              <text x={(tr.x + br.x) / 2 + 2.5} y={(tr.y + br.y) / 2 + 1} fontSize="3" fontWeight="700" fill="#FFFFFF"
                stroke="rgba(0,0,0,.35)" strokeWidth="0.12" fontFamily="var(--font-heading)">{b.pct}%</text>
            </g>
          );
        })}
      </g>
    );
  };

  const shuttle = () => {
    const s = vidShuttleAt(p);
    const cur = vidProject(s.u, s.v);
    const from = vidProject(s.from.u, s.from.v);
    return (
      <g>
        <line x1={from.x} y1={from.y - 4} x2={cur.x} y2={cur.y - 2} stroke="#FFFFFF" strokeWidth="0.4"
          strokeDasharray="1.3 1" opacity="0.7" />
        <circle cx={cur.x} cy={cur.y - 2 - Math.sin(Math.PI * s.segP) * 6} r="1" fill="#FFFFFF"
          stroke="rgba(0,0,0,.3)" strokeWidth="0.2" />
      </g>
    );
  };

  return (
    <svg viewBox={'0 0 ' + VID_VB_W + ' ' + VID_VB_H} preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      {ov.zonen ? zones() : null}
      {ov.wege ? trail(VID_TRACK_NEAR, ORANGE) : null}
      {ov.wege ? trail(VID_TRACK_FAR, BLUE) : null}
      {ov.shuttle ? shuttle() : null}
      {ov.tracking ? playerBox(farPos, far.first, BLUE) : null}
      {ov.tracking ? playerBox(nearPos, near.isViewer ? 'Du' : near.first, ORANGE) : null}
    </svg>
  );
}

function VideoAnalysisCard({ theme, anim, video, players, reportId }) {
  const vref = React.useRef(null);
  const [playing, setPlaying] = React.useState(false);
  const [time, setTime] = React.useState(0);
  const [dur, setDur] = React.useState(0);
  const posKey = 'bca-vt-' + reportId;
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

  // rAF render loop while playing (time drives the overlay)
  React.useEffect(() => {
    if (!playing) return;
    let raf;
    const tick = () => {
      const v = vref.current;
      if (v) setTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // persist position (1s granularity)
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

  return (
    <SectionCard theme={theme} title="Video-Analyse" sub={video.label}
      foot="Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage.">
      <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#0A0A0B', lineHeight: 0 }}>
        <video ref={vref} src={video.src} muted playsInline preload="metadata"
          onLoadedMetadata={onLoaded} onEnded={() => setPlaying(false)}
          onTimeUpdate={() => { const v = vref.current; if (v && v.paused) setTime(v.currentTime); }}
          style={{ width: '100%', display: 'block' }}></video>
        <VideoOverlay time={time} players={players} ov={ov} />
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 12 }}>
        <VidToggleChip label="Tracking" on={ov.tracking} onClick={() => toggle('tracking')} />
        <VidToggleChip label="Laufwege" on={ov.wege} onClick={() => toggle('wege')} />
        <VidToggleChip label="Shuttle" on={ov.shuttle} onClick={() => toggle('shuttle')} />
        <VidToggleChip label="Zonen" on={ov.zonen} onClick={() => toggle('zonen')} />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
        <button type="button" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Abspielen'} style={{
          width: 38, height: 38, borderRadius: 19, border: '1px solid #C66C27', cursor: 'pointer',
          background: '#FA8832', color: '#1A0500', display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 3px 0 #C66C27', flex: 'none'
        }}>
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="5" height="16" rx="1.2"></rect><rect x="14" y="4" width="5" height="16" rx="1.2"></rect></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.8v14.4c0 .9 1 1.5 1.8 1L20 13c.8-.5.8-1.6 0-2.1L8.8 3.9C8 3.4 7 4 7 4.8z"></path></svg>
          )}
        </button>
        <input type="range" min="0" max={dur || 0} step="0.1" value={time}
          onChange={(e) => seek(parseFloat(e.target.value))} className="bca-range"
          style={{ flex: 1, accentColor: '#FA8832' }} />
        <span style={{ fontSize: 11.5, color: theme.sub, fontVariantNumeric: 'tabular-nums', fontFamily: 'var(--font-mono)', flex: 'none' }}>
          {vidFmtTime(time)} / {vidFmtTime(dur)}
        </span>
      </div>
    </SectionCard>
  );
}

Object.assign(window, { VideoAnalysisCard, VideoOverlay, VidToggleChip, vidFmtTime, VID_LOOP });
