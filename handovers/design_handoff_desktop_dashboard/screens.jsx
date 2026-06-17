// screens.jsx — THEMES + MatchListScreen + ReportScreen
// Depends on viz.jsx (CountUp, Reveal, charts, ICONS) + mock-data.js (window.BCA)

const THEMES = {
  A: {
    key: 'A', name: 'Vereinsfarben', pageBg: '#FEF4EA', ink: '#0A0A0B',
    sub: 'hsl(215.4 16.3% 46.9%)', accent: '#FA8832', accentSoft: 'rgba(250,136,50,.42)',
    heat: 'blobs', heatDark: false, timelineStyle: 'bars',
    blurb: 'Strikt nach Markenbuch: Beige, Weiß, Orange, Bubble-Optik, Icon-Chips. Fühlt sich sofort nach Vereins-Website an.'
  },
  B: {
    key: 'B', name: 'Magazin', pageBg: '#FFFFFF', ink: '#0A0A0B',
    sub: 'hsl(215.4 16.3% 46.9%)', accent: '#FA8832', accentSoft: 'rgba(10,10,11,.22)',
    heat: 'raster', heatDark: false, timelineStyle: 'bars',
    blurb: 'Editorial und athletisch: riesige Ziffern, Haarlinien statt Karten, Orange nur als Signal. Wie ein Sportmagazin-Datenblatt.'
  },
  C: {
    key: 'C', name: 'Matchcenter', pageBg: '#FFFFFF', ink: '#0A0A0B',
    sub: 'hsl(215.4 16.3% 46.9%)', accent: '#FA8832', accentSoft: 'rgba(250,136,50,.42)',
    heat: 'blobs', heatDark: true, timelineStyle: 'bars',
    blurb: 'TV-Übertragungs-Gefühl: dunkles Scoreboard als Hero, leuchtende Heatmap, kräftige Karten. Der belohnendste Einstieg.'
  },
  D: {
    key: 'D', name: 'Protokoll', pageBg: '#FEF4EA', ink: '#0A0A0B',
    sub: 'hsl(215.4 16.3% 46.9%)', accent: '#FA8832', accentSoft: 'rgba(10,10,11,.3)',
    heat: 'punkte', heatDark: false, timelineStyle: 'line',
    blurb: 'Ruhig, druckartig, maximale Daten-Tinte: Monospace-Ziffern, Punktraster-Heatmap, gepunktete Linien. Für Zahlenmenschen.'
  }
};

const HAIR = '1px solid hsl(214.3 31.8% 91.4%)';

function StatusBar({ theme, onDark }) {
  const c = onDark ? '#FFFFFF' : theme.ink;
  return (
    <div style={{ height: 42, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 22px', color: c }}>
      <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: 0.2 }}>9:41</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <svg width="17" height="11" viewBox="0 0 17 11" fill={c}><rect x="0" y="7" width="3" height="4" rx="0.8"></rect><rect x="4.5" y="5" width="3" height="6" rx="0.8"></rect><rect x="9" y="2.5" width="3" height="8.5" rx="0.8"></rect><rect x="13.5" y="0" width="3" height="11" rx="0.8"></rect></svg>
        <svg width="16" height="11" viewBox="0 0 16 11" fill="none" stroke={c} strokeWidth="1.6" strokeLinecap="round"><path d="M2 4.2a9 9 0 0 1 12 0"></path><path d="M4.3 6.7a5.6 5.6 0 0 1 7.4 0"></path><circle cx="8" cy="9.3" r="1.2" fill={c} stroke="none"></circle></svg>
        <svg width="25" height="12" viewBox="0 0 25 12" fill="none"><rect x="0.5" y="0.5" width="21" height="11" rx="3" stroke={c} opacity="0.5"></rect><rect x="2" y="2" width="15" height="8" rx="1.6" fill={c}></rect><path d="M23 4v4a2.2 2.2 0 0 0 0-4z" fill={c} opacity="0.5"></path></svg>
      </div>
    </div>
  );
}

function GhostBtn({ theme, onDark, onClick, icon, label }) {
  return (
    <button type="button" onClick={onClick} aria-label={label} className="bca-ghost" style={{
      width: 34, height: 34, borderRadius: 17, display: 'flex', alignItems: 'center', justifyContent: 'center',
      border: onDark ? '1px solid rgba(255,255,255,.3)' : HAIR,
      background: onDark ? 'rgba(255,255,255,.08)' : '#FFFFFF',
      color: onDark ? '#FFFFFF' : theme.ink, cursor: 'pointer', padding: 0, flex: 'none'
    }}>
      <Icon d={icon} size={16} />
    </button>
  );
}

function ResultChip({ result, line, theme, small }) {
  const win = result === 'S';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: small ? '2px 8px' : '3px 10px', borderRadius: 99,
      fontSize: small ? 11 : 12, fontWeight: 700, whiteSpace: 'nowrap',
      background: win ? 'rgba(250,136,50,.16)' : 'rgba(10,10,11,.07)',
      color: win ? '#C66C27' : theme.sub
    }}>{line}</span>
  );
}

function DisciplineChip({ text, theme }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 99, fontSize: 11, fontWeight: 600,
      background: 'rgba(10,10,11,.06)', color: theme.sub
    }}>{text}</span>
  );
}

// ── Section container, themed ──────────────────────────────────
function SectionCard({ theme, title, sub, children, foot }) {
  const k = theme.key;
  if (k === 'B') {
    return (
      <section style={{ borderTop: '2px solid #0A0A0B', padding: '14px 0 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <h3 style={{ margin: 0, fontSize: 11.5, letterSpacing: '0.14em', textTransform: 'uppercase', fontFamily: 'var(--font-body)', fontWeight: 700 }}>{title}</h3>
          {sub ? <span style={{ fontSize: 11.5, color: theme.sub }}>{sub}</span> : null}
        </div>
        <div style={{ marginTop: 14 }}>{children}</div>
        {foot ? <p style={{ margin: '12px 0 0', fontSize: 11.5, color: theme.sub }}>{foot}</p> : null}
      </section>
    );
  }
  if (k === 'D') {
    return (
      <section style={{ border: '1px solid rgba(10,10,11,.22)', borderRadius: 6, padding: '13px 14px 16px', background: 'rgba(255,255,255,.45)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <h3 style={{ margin: 0, fontSize: 10.5, letterSpacing: '0.18em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{title}</h3>
          {sub ? <span style={{ fontSize: 10.5, color: theme.sub, fontFamily: 'var(--font-mono)' }}>{sub}</span> : null}
        </div>
        <div style={{ marginTop: 13 }}>{children}</div>
        {foot ? <p style={{ margin: '11px 0 0', fontSize: 11, color: theme.sub, fontFamily: 'var(--font-mono)' }}>{foot}</p> : null}
      </section>
    );
  }
  // A + C: white card
  return (
    <section style={{
      background: '#FFFFFF', border: HAIR, borderRadius: k === 'C' ? 14 : 12,
      boxShadow: k === 'C' ? '0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.07)' : 'var(--shadow-sm)',
      padding: '15px 16px 18px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontFamily: 'var(--font-heading)', fontWeight: 700, whiteSpace: 'nowrap' }}>{title}</h3>
        {sub ? <span style={{ fontSize: 11.5, color: theme.sub, textAlign: 'right' }}>{sub}</span> : null}
      </div>
      <div style={{ marginTop: 13 }}>{children}</div>
      {foot ? <p style={{ margin: '12px 0 0', fontSize: 11.5, color: theme.sub }}>{foot}</p> : null}
    </section>
  );
}

// ── Key numbers (2×2 / rows) ───────────────────────────────────
function KeyStats({ theme, stats, anim }) {
  const k = theme.key;
  const items = [
    { label: 'Distanz', value: stats.distance_m / 1000, dec: 2, unit: 'km', icon: ICONS.route },
    { label: 'Sprints', value: stats.sprints, dec: 0, unit: '', icon: ICONS.bolt },
    { label: 'Ø Tempo', value: stats.avg_kmh, dec: 1, unit: 'km/h', icon: ICONS.gauge },
    { label: 'Top-Tempo', value: stats.top_kmh, dec: 1, unit: 'km/h', icon: ICONS.clock }
  ];
  if (k === 'D') {
    return (
      <SectionCard theme={theme} title="Kennzahlen" sub="aktive Spielzeit">
        <div style={{ display: 'grid', gap: 8 }}>
          {items.map((it) => (
            <div key={it.label} style={{ display: 'flex', alignItems: 'baseline', gap: 8, fontFamily: 'var(--font-mono)' }}>
              <span style={{ fontSize: 12.5, color: theme.sub }}>{it.label}</span>
              <span style={{ flex: 1, borderBottom: '1px dotted rgba(10,10,11,.35)', transform: 'translateY(-3px)' }}></span>
              <strong style={{ fontSize: 15, fontVariantNumeric: 'tabular-nums' }}>
                <CountUp value={it.value} decimals={it.dec} anim={anim} />{it.unit ? '\u2009' + it.unit : ''}
              </strong>
            </div>
          ))}
        </div>
      </SectionCard>
    );
  }
  if (k === 'B') {
    return (
      <section style={{ borderTop: '2px solid #0A0A0B', display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
        {items.map((it, i) => (
          <div key={it.label} style={{
            padding: '14px 0 16px', paddingLeft: i % 2 === 1 ? 18 : 0,
            borderLeft: i % 2 === 1 ? '1px solid #E5E2DD' : 'none',
            borderTop: i > 1 ? '1px solid #E5E2DD' : 'none'
          }}>
            <div style={{ fontSize: 10.5, letterSpacing: '0.14em', textTransform: 'uppercase', color: theme.sub, fontWeight: 600 }}>{it.label}</div>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 38, lineHeight: 1.1, letterSpacing: '-0.02em', marginTop: 2 }}>
              <CountUp value={it.value} decimals={it.dec} anim={anim} />
              <span style={{ fontSize: 14, fontWeight: 600, color: theme.sub, marginLeft: 4 }}>{it.unit}</span>
            </div>
          </div>
        ))}
      </section>
    );
  }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      {items.map((it) => (
        <div key={it.label} style={{
          background: '#FFFFFF', border: HAIR, borderRadius: k === 'C' ? 14 : 12,
          boxShadow: k === 'C' ? '0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.07)' : 'var(--shadow-sm)',
          padding: '13px 14px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 28, height: 28, borderRadius: 9, background: 'rgba(250,136,50,.13)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#C66C27' }}>
              <Icon d={it.icon} size={15} />
            </span>
            <span style={{ fontSize: 11.5, color: theme.sub, fontWeight: 500, whiteSpace: 'nowrap' }}>{it.label}</span>
          </div>
          <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 24, marginTop: 8, letterSpacing: '-0.01em' }}>
            <CountUp value={it.value} decimals={it.dec} anim={anim} />
            <span style={{ fontSize: 12.5, fontWeight: 600, color: theme.sub, marginLeft: 4 }}>{it.unit}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Player tabs ────────────────────────────────────────────────
function PlayerTabs({ theme, players, tab, setTab }) {
  const k = theme.key;
  const label = (p) => p.isViewer ? 'Du' : p.first;
  if (k === 'B') {
    return (
      <div style={{ display: 'flex', gap: 22, borderBottom: '1px solid #E5E2DD' }}>
        {players.map((p, i) => (
          <button key={p.name} type="button" onClick={() => setTab(i)} className="bca-tab" style={{
            background: 'none', border: 'none', padding: '8px 0 10px', cursor: 'pointer',
            fontSize: 13.5, fontWeight: tab === i ? 700 : 500, fontFamily: 'var(--font-body)',
            color: tab === i ? theme.ink : theme.sub,
            boxShadow: tab === i ? 'inset 0 -2px 0 #FA8832' : 'none'
          }}>{label(p)}</button>
        ))}
      </div>
    );
  }
  if (k === 'D') {
    return (
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontFamily: 'var(--font-mono)' }}>
        {players.map((p, i) => (
          <button key={p.name} type="button" onClick={() => setTab(i)} className="bca-tab" style={{
            background: tab === i ? '#0A0A0B' : 'transparent',
            color: tab === i ? '#FEF4EA' : theme.sub,
            border: '1px solid ' + (tab === i ? '#0A0A0B' : 'rgba(10,10,11,.3)'),
            borderRadius: 4, padding: '5px 12px', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit'
          }}>{label(p)}</button>
        ))}
      </div>
    );
  }
  // A + C: segmented pills
  return (
    <div style={{
      display: 'flex', gap: 4, padding: 4, borderRadius: 12,
      background: k === 'C' ? 'rgba(10,10,11,.05)' : 'rgba(10,10,11,.05)', border: HAIR
    }}>
      {players.map((p, i) => (
        <button key={p.name} type="button" onClick={() => setTab(i)} className="bca-tab" style={{
          flex: 1, border: 'none', borderRadius: 9, padding: '8px 4px', cursor: 'pointer',
          fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-body)',
          background: tab === i ? (k === 'C' ? '#0A0A0B' : '#FFFFFF') : 'transparent',
          color: tab === i ? (k === 'C' ? '#FFFFFF' : theme.ink) : theme.sub,
          boxShadow: tab === i && k === 'A' ? 'var(--shadow-sm)' : 'none',
          transition: 'background .2s ease, color .2s ease'
        }}>{label(p)}</button>
      ))}
    </div>
  );
}

// ── Report hero, themed ────────────────────────────────────────
function ReportHero({ theme, report, player, anim, onBack, onReplay }) {
  const k = theme.key;
  const m = report.meta;
  const first = window.BCA.viewerFirst;
  const headline = m.result === 'S' ? 'Stark gespielt, ' + first + '.' : 'Gut gekämpft, ' + first + '.';
  const backRow = onBack ? (
    <div style={{ marginBottom: 10 }}>
      <GhostBtn theme={theme} onClick={onBack} icon={ICONS.arrowL} label="Zurück" />
    </div>
  ) : null;

  if (k === 'B') {
    return (
      <header style={{ padding: '4px 18px 18px' }}>
        {backRow}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 10.5, letterSpacing: '0.16em', textTransform: 'uppercase', fontWeight: 700, color: theme.sub }}>
            Spielbericht · {m.dateLabel}
          </span>
          <GhostBtn theme={theme} onClick={onReplay} icon={ICONS.replay} label="Animationen wiederholen" />
        </div>
        <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 68, lineHeight: 1, letterSpacing: '-0.03em', marginTop: 12 }}>
          <CountUp value={player.stats.distance_m / 1000} decimals={2} anim={anim} />
          <span style={{ fontSize: 20, fontWeight: 600, color: theme.sub, marginLeft: 6 }}>km</span>
        </div>
        <p style={{ margin: '8px 0 0', fontSize: 13.5, color: theme.sub }}>
          gelaufen in {m.durationMin} Minuten — {player.isViewer ? 'von dir' : 'von ' + player.first}, {m.discipline} · {m.score}
        </p>
        <div style={{ marginTop: 10 }}>
          <ResultChip result={m.result} line={m.resultLine} theme={theme} />
        </div>
      </header>
    );
  }

  if (k === 'C') {
    const sideName = (side) => report.pairs
      ? report.pairs.find((p) => p.side === side).label
      : report.players.find((p) => p.side === side).first + ' ' + report.players.find((p) => p.side === side).name.split(' ')[1];
    return (
      <header style={{ padding: '4px 14px 0' }}>
        {backRow}
        <div className="on-dark" style={{ background: '#0A0A0B', borderRadius: 16, padding: '16px 16px 18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 11, letterSpacing: '0.14em', textTransform: 'uppercase', fontWeight: 700, color: '#FA8832' }}>
              {m.discipline} · Feld {m.court}
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11.5, color: 'rgba(255,255,255,.55)' }}>{m.dateLabel}</span>
              <GhostBtn theme={theme} onDark onClick={onReplay} icon={ICONS.replay} label="Animationen wiederholen" />
            </span>
          </div>
          <div style={{ marginTop: 14, display: 'grid', gap: 9 }}>
            {['A', 'B'].map((side, si) => (
              <div key={side} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  flex: 1, fontFamily: 'var(--font-heading)', fontWeight: side === 'A' ? 700 : 500, fontSize: 15.5,
                  color: side === 'A' ? '#FFFFFF' : 'rgba(255,255,255,.65)'
                }}>{sideName(side)}</span>
                <span style={{ display: 'flex', gap: 6 }}>
                  {m.games.map((g, gi) => {
                    const v = side === 'A' ? g[0] : g[1];
                    const won = side === 'A' ? g[0] > g[1] : g[1] > g[0];
                    return (
                      <span key={gi} style={{
                        width: 30, height: 26, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 14, fontVariantNumeric: 'tabular-nums',
                        background: won ? '#FA8832' : 'rgba(255,255,255,.1)',
                        color: won ? '#1A0500' : 'rgba(255,255,255,.75)'
                      }}>{v}</span>
                    );
                  })}
                </span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{
              padding: '3px 10px', borderRadius: 99, fontSize: 12, fontWeight: 700,
              background: m.result === 'S' ? '#FA8832' : 'rgba(255,255,255,.12)',
              color: m.result === 'S' ? '#1A0500' : 'rgba(255,255,255,.75)'
            }}>{m.resultLine}</span>
            <span style={{ fontSize: 11.5, color: 'rgba(255,255,255,.45)' }}>{m.durationMin} Min Spielzeit</span>
          </div>
        </div>
      </header>
    );
  }

  if (k === 'D') {
    const rows = [
      ['Datum', m.weekday + ', ' + m.dateLabel],
      ['Disziplin', m.discipline + ' · Feld ' + m.court],
      ['Ergebnis', m.score],
      ['Wertung', m.resultLine],
      ['Dauer', m.durationMin + ' min (' + m.restMin + ' min Pause)']
    ];
    return (
      <header style={{ padding: '4px 16px 0' }}>
        {backRow}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 11, letterSpacing: '0.2em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>Spielbericht</span>
          <GhostBtn theme={theme} onClick={onReplay} icon={ICONS.replay} label="Animationen wiederholen" />
        </div>
        <div style={{ marginTop: 12, display: 'grid', gap: 6, fontFamily: 'var(--font-mono)' }}>
          {rows.map((r) => (
            <div key={r[0]} style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontSize: 12, color: theme.sub, width: 70, flex: 'none' }}>{r[0]}</span>
              <span style={{ flex: 1, borderBottom: '1px dotted rgba(10,10,11,.3)', transform: 'translateY(-3px)' }}></span>
              <span style={{ fontSize: 12.5, fontWeight: 600, textAlign: 'right' }}>{r[1]}</span>
            </div>
          ))}
        </div>
      </header>
    );
  }

  // A — warm brand greeting
  return (
    <header style={{ padding: '4px 18px 0' }}>
      {backRow}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
        <div>
          <div className="eyebrow" style={{ fontSize: 12, marginBottom: 4 }}>Spielbericht</div>
          <h2 style={{ margin: 0, fontSize: 27, lineHeight: 1.08, fontFamily: 'var(--font-heading)', letterSpacing: '-0.01em' }}>{headline}</h2>
        </div>
        <GhostBtn theme={theme} onClick={onReplay} icon={ICONS.replay} label="Animationen wiederholen" />
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 13, color: theme.sub }}>
        {m.weekday}, {m.dateLabel} · {m.discipline} · Feld {m.court}
      </p>
      <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <ResultChip result={m.result} line={m.resultLine} theme={theme} />
        <span style={{ fontSize: 13, color: theme.sub, fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
      </div>
    </header>
  );
}

// ── Full report screen ─────────────────────────────────────────
function ReportScreen({ theme, anim, report, heatOverride = 'auto', onBack }) {
  const [tab, setTab] = React.useState(0);
  const [run, setRun] = React.useState(0);
  const k = theme.key;
  const p = report.players[tab];
  const s = p.stats;
  const heatStyle = heatOverride !== 'auto' ? heatOverride : theme.heat;
  const totalShots = s.shots.reduce((a, x) => a + x[1], 0);
  const who = p.isViewer ? 'du' : p.first;
  const pad = k === 'B' ? '0 18px' : k === 'D' ? '0 16px' : '0 14px';
  const gap = k === 'B' ? 0 : k === 'D' ? 10 : 12;

  return (
    <div data-screen-label={theme.name + ' · Bericht ' + report.meta.discipline} style={{
      width: 390, minHeight: '100%', background: theme.pageBg, color: theme.ink,
      fontFamily: 'var(--font-body)', paddingBottom: 28, boxSizing: 'border-box'
    }}>
      <StatusBar theme={theme} />
      <div key={run}>
        <Reveal i={0} anim={anim}>
          <ReportHero theme={theme} report={report} player={p} anim={anim} onBack={onBack} onReplay={() => setRun(run + 1)} />
        </Reveal>

        <div style={{ padding: pad, marginTop: 16 }}>
          <Reveal i={1} anim={anim}>
            <PlayerTabs theme={theme} players={report.players} tab={tab} setTab={setTab} />
          </Reveal>
        </div>

        <div key={'tab' + tab} style={{ padding: pad, marginTop: k === 'B' ? 14 : 14, display: 'grid', gap: gap }}>
          <Reveal i={2} anim={anim}><KeyStats theme={theme} stats={s} anim={anim} /></Reveal>

          {report.video && window.VideoAnalysisCard ? (
            <Reveal i={3} anim={anim}>
              <VideoAnalysisCard theme={theme} anim={anim} video={report.video}
                players={report.players} reportId={report.meta.id} />
            </Reveal>
          ) : null}

          <Reveal i={3} anim={anim}>
            <SectionCard theme={theme} title="Feldabdeckung" sub={p.isViewer ? 'wo du gestanden hast' : 'wo ' + p.first + ' gestanden hat'}>
              <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0 0' }}>
                <CourtHeatmap spots={s.heat} style={heatStyle} dark={theme.heatDark} width={222} anim={anim} />
              </div>
              <HeatLegend theme={theme} />
            </SectionCard>
          </Reveal>

          <Reveal i={4} anim={anim}>
            <SectionCard theme={theme} title="Spielfeld-Zonen" sub="Anteil aktiver Spielzeit">
              <ZoneBlock zones={s.zones} theme={theme} anim={anim} />
            </SectionCard>
          </Reveal>

          <Reveal i={5} anim={anim}>
            <SectionCard theme={theme} title="Aktivität" sub="Distanz je 5 Minuten">
              <TimelineChart points={s.timeline} theme={theme} anim={anim} />
            </SectionCard>
          </Reveal>

          <Reveal i={6} anim={anim}>
            <SectionCard theme={theme} title="Schlagverteilung" sub={deNum(totalShots) + ' Schläge'}
              foot={s.fastest ? 'Schnellster Smash ≈ ' + deNum(s.fastest) + ' km/h — Schätzung aus der Shuttle-Verfolgung.' : null}>
              <ShotMix shots={s.shots} theme={theme} anim={anim} />
            </SectionCard>
          </Reveal>

          <Reveal i={7} anim={anim}>
            <SectionCard theme={theme} title="Ballwechsel" sub={report.rallies.endLine ? null : null}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, textAlign: 'center' }}>
                {[
                  [report.rallies.count, 0, 'gesamt'],
                  [report.rallies.avgShots, 1, 'Ø Schläge'],
                  [report.rallies.longShots, 0, 'längster']
                ].map((it, i) => (
                  <div key={it[2]}>
                    <div style={{ fontFamily: theme.key === 'D' ? 'var(--font-mono)' : 'var(--font-heading)', fontWeight: 700, fontSize: theme.key === 'B' ? 30 : 22 }}>
                      <CountUp value={it[0]} decimals={it[1]} anim={anim} />
                    </div>
                    <div style={{ fontSize: 11.5, color: theme.sub, marginTop: 2 }}>{it[2]}</div>
                  </div>
                ))}
              </div>
              <p style={{ margin: '12px 0 0', fontSize: 12.5, color: theme.sub, textAlign: 'center' }}>
                Längster Ballwechsel: {report.rallies.longShots} Schläge in {report.rallies.longSec}&thinsp;s. {report.rallies.endLine}
              </p>
            </SectionCard>
          </Reveal>

          {report.pairs ? (
            <Reveal i={8} anim={anim}>
              <SectionCard theme={theme} title="Doppel-Formation" sub="Anteil aktiver Spielzeit">
                <FormationBlock pairs={report.pairs} theme={theme} anim={anim} />
              </SectionCard>
            </Reveal>
          ) : null}

          <Reveal i={report.pairs ? 9 : 8} anim={anim}>
            <footer style={{ textAlign: 'center', padding: '14px 10px 4px' }}>
              <img src="assets/logo-mark.svg" alt="BC Rheinbach" style={{ height: 26, opacity: 0.85 }} />
              <p style={{ margin: '8px 0 0', fontSize: 11, color: theme.sub, lineHeight: 1.55 }}>
                Automatisch aus der Videoaufnahme ausgewertet.<br />
                Du möchtest deine Daten löschen? Schreib uns kurz.
              </p>
            </footer>
          </Reveal>
        </div>
      </div>
    </div>
  );
}

// ── Match list screen ──────────────────────────────────────────
function MatchListScreen({ theme, anim, onOpen }) {
  const k = theme.key;
  const matches = window.BCA.matches;
  const open = (m) => { if (onOpen) onOpen(m); };
  const cursor = onOpen ? 'pointer' : 'default';

  const header = (() => {
    if (k === 'B') {
      return (
        <header style={{ padding: '6px 18px 4px' }}>
          <div style={{ fontSize: 10.5, letterSpacing: '0.16em', textTransform: 'uppercase', fontWeight: 700, color: theme.sub }}>
            BC Rheinbach · Spielanalyse
          </div>
          <h1 style={{ margin: '6px 0 0', fontSize: 46, lineHeight: 1, fontFamily: 'var(--font-heading)', letterSpacing: '-0.025em' }}>Berichte</h1>
          <p style={{ margin: '10px 0 0', fontSize: 13.5, color: theme.sub }}>Hallo Lukas — deine Auswertungen nach jedem Spiel.</p>
        </header>
      );
    }
    if (k === 'C') {
      return (
        <header style={{ padding: '4px 14px 0' }}>
          <div className="on-dark" style={{ background: '#0A0A0B', borderRadius: 16, padding: '16px 16px 17px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <img src="assets/logo-mark.svg" alt="" style={{ height: 30 }} />
              <div>
                <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 18 }}>Matchcenter</div>
                <div style={{ fontSize: 12, color: 'rgba(255,255,255,.55)' }}>BC Rheinbach</div>
              </div>
              <span style={{ marginLeft: 'auto', padding: '3px 10px', borderRadius: 99, background: 'rgba(250,136,50,.18)', color: '#FA8832', fontSize: 11.5, fontWeight: 700 }}>
                {matches.length} Spiele
              </span>
            </div>
            <p style={{ margin: '12px 0 0', fontSize: 13, color: 'rgba(255,255,255,.65)' }}>Hallo Lukas, deine Spiele auf einen Blick.</p>
          </div>
        </header>
      );
    }
    if (k === 'D') {
      return (
        <header style={{ padding: '6px 16px 2px', fontFamily: 'var(--font-mono)' }}>
          <div style={{ fontSize: 10.5, letterSpacing: '0.2em', textTransform: 'uppercase', fontWeight: 700, color: theme.sub }}>
            BC Rheinbach · Spielanalyse
          </div>
          <h1 style={{ margin: '8px 0 0', fontSize: 24, fontFamily: 'var(--font-mono)', fontWeight: 700, letterSpacing: '-0.01em' }}>Spielberichte</h1>
          <p style={{ margin: '8px 0 0', fontSize: 12, color: theme.sub }}>5 ausgewertete Spiele · Stand 12.06.2026</p>
        </header>
      );
    }
    return (
      <header style={{ padding: '6px 18px 4px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <img src="assets/logo-mark.svg" alt="BC Rheinbach" style={{ height: 28 }} />
        </div>
        <h1 style={{ margin: '12px 0 0', fontSize: 30, lineHeight: 1.05, fontFamily: 'var(--font-heading)', letterSpacing: '-0.01em' }}>Spielberichte</h1>
        <p style={{ margin: '8px 0 0', fontSize: 13.5, color: theme.sub }}>Hallo Lukas — deine Auswertungen nach jedem Spiel.</p>
      </header>
    );
  })();

  const row = (m, i) => {
    const names = (
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{m.a}</div>
        <div style={{ fontSize: 12.5, color: theme.sub, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>gegen {m.b}</div>
      </div>
    );
    if (k === 'B') {
      return (
        <div key={m.id} className="bca-row-flat" onClick={() => open(m)} style={{
          display: 'grid', gridTemplateColumns: '64px 1fr auto', gap: 12, alignItems: 'center',
          padding: '14px 18px', borderBottom: '1px solid #E5E2DD', cursor
        }}>
          <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 21, letterSpacing: '-0.01em' }}>{m.dd}</div>
          <div style={{ minWidth: 0 }}>
            {names}
            <div style={{ fontSize: 10.5, letterSpacing: '0.12em', textTransform: 'uppercase', color: theme.sub, fontWeight: 600, marginTop: 4 }}>
              {m.discipline} · {m.score} · <span style={{ color: m.result === 'S' ? '#C66C27' : theme.sub }}>{m.result === 'S' ? 'Sieg' : 'Niederlage'}</span>
            </div>
          </div>
          <span style={{ color: theme.accent }}><Icon d={ICONS.chevR} size={18} /></span>
        </div>
      );
    }
    if (k === 'D') {
      return (
        <div key={m.id} className="bca-row-flat" onClick={() => open(m)} style={{
          padding: '12px 16px', borderBottom: '1px dotted rgba(10,10,11,.35)', fontFamily: 'var(--font-mono)', cursor
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: theme.sub }}>
            <span>{m.dd} · {m.discipline}</span>
            <span style={{ color: m.result === 'S' ? '#C66C27' : theme.sub, fontWeight: 700 }}>{m.result === 'S' ? 'SIEG' : 'NIED.'}</span>
          </div>
          <div style={{ fontSize: 13.5, fontWeight: 600, marginTop: 4 }}>{m.a}</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: theme.sub, marginTop: 2 }}>
            <span>gegen {m.b}</span>
            <span style={{ fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
          </div>
        </div>
      );
    }
    if (k === 'C') {
      return (
        <div key={m.id} className={onOpen ? 'bca-press' : ''} onClick={() => open(m)} style={{
          display: 'grid', gridTemplateColumns: '38px 1fr 18px', gap: 12, alignItems: 'center',
          background: '#FFFFFF', border: HAIR, borderRadius: 14, padding: '13px 14px', cursor,
          boxShadow: '0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.07)'
        }}>
          <span style={{
            width: 38, height: 38, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 15,
            background: m.result === 'S' ? '#FA8832' : 'rgba(10,10,11,.07)',
            color: m.result === 'S' ? '#1A0500' : theme.sub
          }}>{m.result}</span>
          <div style={{ minWidth: 0 }}>
            {names}
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 5, flexWrap: 'wrap' }}>
              <DisciplineChip text={m.discipline} theme={theme} />
              <span style={{ fontSize: 12, color: theme.sub, fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
              {m.isNew ? <span style={{ fontSize: 11, fontWeight: 700, color: '#C66C27' }}>● Neu</span> : null}
            </div>
          </div>
          <span style={{ color: theme.sub }}><Icon d={ICONS.chevR} size={17} /></span>
        </div>
      );
    }
    // A
    return (
      <div key={m.id} className={onOpen ? 'bca-press' : ''} onClick={() => open(m)} style={{
        display: 'grid', gridTemplateColumns: '46px 1fr 18px', gap: 12, alignItems: 'center',
        background: '#FFFFFF', border: HAIR, borderRadius: 12, padding: '13px 14px',
        boxShadow: 'var(--shadow-sm)', cursor, position: 'relative'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 21, lineHeight: 1 }}>{m.day}</div>
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: theme.sub, marginTop: 2 }}>{m.month}</div>
        </div>
        <div style={{ minWidth: 0 }}>
          {names}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 5, flexWrap: 'wrap' }}>
            <DisciplineChip text={m.discipline} theme={theme} />
            <ResultChip result={m.result} line={m.result === 'S' ? 'Sieg' : 'Niederlage'} theme={theme} small />
            <span style={{ fontSize: 12, color: theme.sub, fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
          </div>
        </div>
        <span style={{ color: theme.accent }}><Icon d={ICONS.chevR} size={17} /></span>
        {m.isNew ? (
          <span style={{
            position: 'absolute', top: -7, right: 12, padding: '2px 9px', borderRadius: 99,
            background: '#FA8832', color: '#1A0500', fontSize: 10.5, fontWeight: 700,
            boxShadow: '0 2px 0 #C66C27'
          }}>Neu</span>
        ) : null}
      </div>
    );
  };

  return (
    <div data-screen-label={theme.name + ' · Spielliste'} style={{
      width: 390, minHeight: '100%', background: theme.pageBg, color: theme.ink,
      fontFamily: 'var(--font-body)', paddingBottom: 24, boxSizing: 'border-box'
    }}>
      <StatusBar theme={theme} />
      <Reveal i={0} anim={anim}>{header}</Reveal>
      <div style={{
        marginTop: k === 'B' ? 10 : 16,
        padding: k === 'B' || k === 'D' ? 0 : '0 14px',
        display: 'grid', gap: k === 'B' || k === 'D' ? 0 : 10
      }}>
        {matches.map((m, i) => <Reveal key={m.id} i={i + 1} anim={anim}>{row(m, i)}</Reveal>)}
      </div>
      <Reveal i={7} anim={anim}>
        <p style={{ textAlign: 'center', fontSize: 11, color: theme.sub, margin: '20px 24px 0', lineHeight: 1.55 }}>
          Automatisch aus Videoaufnahmen ausgewertet.<br />Nur für Mitglieder sichtbar.
        </p>
      </Reveal>
    </div>
  );
}

Object.assign(window, { THEMES, StatusBar, MatchListScreen, ReportScreen, SectionCard, KeyStats, PlayerTabs, ReportHero, GhostBtn, ResultChip });
