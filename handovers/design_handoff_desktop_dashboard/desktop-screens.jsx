// desktop-screens.jsx — 4 Desktop-Layout-Richtungen für die Spielanalyse
// (alle in Richtung A · Vereinsfarben). Depends on: viz.jsx, screens.jsx,
// video.jsx, desktop-shared.jsx, mock-data.js

// ───────────────────────────────────────────────────────────────
// 1 · MATCHCENTER — Split-View (Liste links, Bericht rechts)
// ───────────────────────────────────────────────────────────────
function DeskSplitScreen({ anim, heat, report }) {
  const [tab, setTab] = React.useState(0);
  const [run, setRun] = React.useState(0);
  const p = report.players[tab], s = p.stats, m = report.meta;
  const matches = window.BCA.matches;
  const first = window.BCA.viewerFirst;

  return (
    <div data-screen-label="Desktop · Matchcenter Split-View" style={{
      width: 1440, minHeight: '100%', display: 'flex', alignItems: 'stretch',
      background: '#FEF4EA', color: DK_INK, fontFamily: 'var(--font-body)', boxSizing: 'border-box'
    }}>
      <aside style={{ width: 330, flex: 'none', borderRight: DK_HAIR, padding: '26px 16px 20px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 6px' }}>
          <img src="assets/logo-mark.svg" alt="BC Rheinbach" style={{ height: 34 }} />
          <div>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 17, lineHeight: 1.1 }}>Spielanalyse</div>
            <div style={{ fontSize: 12, color: DK_SUB }}>BC Rheinbach</div>
          </div>
        </div>
        <p style={{ margin: '20px 6px 10px', fontSize: 12.5, color: DK_SUB }}>Hallo Lukas — deine letzten Spiele.</p>
        <div style={{ display: 'grid', gap: 8 }}>
          {matches.map((x) => {
            const sel = x.id === m.id;
            return (
              <div key={x.id} className={sel ? '' : 'bca-row-flat'} style={{
                display: 'grid', gridTemplateColumns: '40px 1fr', gap: 10, alignItems: 'center',
                padding: '11px 12px', borderRadius: 12, cursor: 'default',
                background: sel ? '#FFFFFF' : 'transparent',
                border: sel ? DK_HAIR : '1px solid transparent',
                boxShadow: sel ? 'var(--shadow-sm)' : 'none'
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 19, lineHeight: 1, color: sel ? DK_ORANGE_DARK : DK_INK }}>{x.day}</div>
                  <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: DK_SUB, marginTop: 2 }}>{x.month}</div>
                </div>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>gegen {x.b}</div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 3, fontSize: 11.5, color: DK_SUB }}>
                    <span>{x.discipline}</span>
                    <span style={{ color: x.result === 'S' ? DK_ORANGE_DARK : DK_SUB, fontWeight: 700 }}>{x.result === 'S' ? 'Sieg' : 'Niederlage'}</span>
                    {x.isNew ? <span style={{ marginLeft: 'auto', padding: '1px 7px', borderRadius: 99, background: DK_ORANGE, color: '#1A0500', fontSize: 10, fontWeight: 700 }}>Neu</span> : null}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <p style={{ margin: 'auto 6px 0', paddingTop: 18, fontSize: 11, color: DK_SUB, lineHeight: 1.55 }}>
          Automatisch aus Videoaufnahmen ausgewertet.<br />Nur für Mitglieder sichtbar.
        </p>
      </aside>

      <main key={run} style={{ flex: 1, minWidth: 0, padding: '30px 36px 34px', boxSizing: 'border-box' }}>
        <Reveal i={0} anim={anim}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 24 }}>
            <div>
              <div className="eyebrow" style={{ fontSize: 12, marginBottom: 4 }}>Spielbericht</div>
              <h1 style={{ margin: 0, fontSize: 34, lineHeight: 1.05, fontFamily: 'var(--font-heading)', letterSpacing: '-0.01em' }}>
                {m.result === 'S' ? 'Stark gespielt, ' + first + '.' : 'Gut gekämpft, ' + first + '.'}
              </h1>
              <p style={{ margin: '8px 0 0', fontSize: 13.5, color: DK_SUB }}>
                {m.weekday}, {m.dateLabel} · {m.discipline} · Feld {m.court}
              </p>
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 10 }}>
                <ResultChip result={m.result} line={m.resultLine} theme={THEMES.A} />
                <span style={{ fontSize: 13.5, color: DK_SUB, fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
                <span style={{ fontSize: 13.5, color: DK_SUB }}>· {m.durationMin} Min Spielzeit</span>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 'none' }}>
              <div style={{ width: 250 }}>
                <PlayerTabs theme={THEMES.A} players={report.players} tab={tab} setTab={setTab} />
              </div>
              <GhostBtn theme={THEMES.A} onClick={() => setRun(run + 1)} icon={ICONS.replay} label="Animationen wiederholen" />
            </div>
          </div>
        </Reveal>

        <div style={{ marginTop: 22, display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 14 }}>
          {dkItems(s).map((it, i) => (
            <div key={it.label} style={{ gridColumn: 'span 3', display: 'grid' }}>
              <Reveal i={1 + i} anim={anim} style={{ display: 'grid' }}><DeskKpiTile it={it} anim={anim} /></Reveal>
            </div>
          ))}

          {report.video ? (
            <div style={{ gridColumn: 'span 8', display: 'grid' }}>
              <Reveal i={3} anim={anim} style={{ display: 'grid' }}>
                <DeskVideoCard video={report.video} players={report.players} posId={m.id} style={{ height: '100%' }} />
              </Reveal>
            </div>
          ) : null}

          <div style={{ gridColumn: report.video ? 'span 4' : 'span 12', display: 'grid', gap: 14, alignContent: 'start' }}>
            <Reveal i={4} anim={anim}>
              <DeskCard title="Feldabdeckung" sub={p.isViewer ? 'wo du gestanden hast' : 'wo ' + p.first + ' gestanden hat'}>
                <div style={{ display: 'flex', justifyContent: 'center', padding: '2px 0 0' }}>
                  <CourtHeatmap spots={s.heat} style={heat} width={236} anim={anim} />
                </div>
                <HeatLegend theme={THEMES.A} />
              </DeskCard>
            </Reveal>
            <Reveal i={5} anim={anim}>
              <DeskCard title="Spielfeld-Zonen" sub="Anteil aktiver Spielzeit">
                <ZoneBlock zones={s.zones} theme={THEMES.A} anim={anim} />
              </DeskCard>
            </Reveal>
          </div>

          <div style={{ gridColumn: 'span 7', display: 'grid' }}>
            <Reveal i={6} anim={anim} style={{ display: 'grid' }}>
              <DeskCard title="Aktivität" sub="Distanz je 5 Minuten" style={{ height: '100%' }}>
                <TimelineChart points={s.timeline} theme={THEMES.A} anim={anim} />
              </DeskCard>
            </Reveal>
          </div>
          <div style={{ gridColumn: 'span 5', display: 'grid' }}>
            <Reveal i={7} anim={anim} style={{ display: 'grid' }}>
              <DeskCard title="Schlagverteilung" sub={deNum(s.shots.reduce((a, x) => a + x[1], 0)) + ' Schläge'} style={{ height: '100%' }}
                foot={s.fastest ? 'Schnellster Smash ≈ ' + deNum(s.fastest) + ' km/h — Schätzung aus der Shuttle-Verfolgung.' : null}>
                <ShotMix shots={s.shots} theme={THEMES.A} anim={anim} />
              </DeskCard>
            </Reveal>
          </div>

          <div style={{ gridColumn: 'span 12' }}>
            <Reveal i={8} anim={anim}>
              <DeskCard title="Ballwechsel">
                <DeskRallies rallies={report.rallies} anim={anim} horizontal />
              </DeskCard>
            </Reveal>
          </div>

          {report.pairs ? (
            <div style={{ gridColumn: 'span 12' }}>
              <Reveal i={9} anim={anim}>
                <DeskCard title="Doppel-Formation" sub="Anteil aktiver Spielzeit">
                  <FormationBlock pairs={report.pairs} theme={THEMES.A} anim={anim} />
                </DeskCard>
              </Reveal>
            </div>
          ) : null}
        </div>

        <footer style={{ textAlign: 'center', padding: '26px 10px 0' }}>
          <img src="assets/logo-mark.svg" alt="BC Rheinbach" style={{ height: 26, opacity: 0.85 }} />
          <p style={{ margin: '8px 0 0', fontSize: 11, color: DK_SUB, lineHeight: 1.55 }}>
            Automatisch aus der Videoaufnahme ausgewertet. Du möchtest deine Daten löschen? Schreib uns kurz.
          </p>
        </footer>
      </main>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// 2 · VEREINSPORTAL — Editorial-Seite im Stil von bc-rheinbach.de
// ───────────────────────────────────────────────────────────────
function DeskPortalScreen({ anim, heat, report }) {
  const [tab, setTab] = React.useState(0);
  const p = report.players[tab], s = p.stats, m = report.meta;
  const first = window.BCA.viewerFirst;

  const NavLink = ({ children, active }) => (
    <a className="dk-navlink" style={{
      fontSize: 15, fontWeight: active ? 700 : 500,
      color: active ? DK_ORANGE_DARK : undefined,
      textDecoration: 'none', cursor: 'pointer'
    }}>{children}</a>
  );
  const Band = ({ bg, children, pad }) => (
    <section style={{ background: bg, padding: pad || '60px 5%' }}>
      <div style={{ maxWidth: 1120, margin: '0 auto' }}>{children}</div>
    </section>
  );
  const SectionHead = ({ eyebrow, title, sub, center }) => (
    <div style={{ textAlign: center ? 'center' : 'left' }}>
      <div className="eyebrow" style={{ fontSize: 13 }}>{eyebrow}</div>
      <h2 style={{ margin: '10px 0 0', fontSize: 42, lineHeight: 1.05, fontFamily: 'var(--font-heading)', letterSpacing: '-0.015em', fontWeight: 700 }}>{title}</h2>
      {sub ? <p style={{ margin: '12px 0 0', fontSize: 15.5, color: DK_SUB, lineHeight: 1.6 }}>{sub}</p> : null}
    </div>
  );

  return (
    <div data-screen-label="Desktop · Vereinsportal" style={{
      width: 1440, minHeight: '100%', background: '#FFFFFF', color: DK_INK,
      fontFamily: 'var(--font-body)', boxSizing: 'border-box'
    }}>
      <header style={{ background: '#FEF4EA', borderBottom: DK_HAIR, padding: '0 5%' }}>
        <div style={{ maxWidth: 1280, margin: '0 auto', height: 76, display: 'flex', alignItems: 'center', gap: 40 }}>
          <img src="assets/logo.svg" style={{ height: 42 }} alt="BC Rheinbach" />
          <nav style={{ display: 'flex', gap: 28 }}>
            <NavLink>Start</NavLink>
            <NavLink active>Spielanalyse</NavLink>
            <NavLink>Training</NavLink>
            <NavLink>Galerie</NavLink>
            <NavLink>Kontakt</NavLink>
          </nav>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ width: 34, height: 34, borderRadius: 17, background: 'rgba(250,136,50,.15)', color: DK_ORANGE_DARK, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-heading)' }}>LB</span>
            <span style={{ fontSize: 14, fontWeight: 600 }}>Lukas</span>
          </div>
        </div>
      </header>

      <Reveal i={0} anim={anim}>
        <Band bg="#FEF4EA" pad="46px 5% 58px">
          <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 64, alignItems: 'center' }}>
            <div>
              <div className="eyebrow" style={{ fontSize: 13 }}>Spielbericht · {m.dateLabel}</div>
              <h1 style={{ margin: '12px 0 0', fontSize: 72, lineHeight: 1.02, fontFamily: 'var(--font-heading)', letterSpacing: '-0.02em', fontWeight: 700 }}>
                {m.result === 'S' ? 'Stark gespielt, ' + first + '.' : 'Gut gekämpft, ' + first + '.'}
              </h1>
              <p style={{ margin: '18px 0 0', fontSize: 17, color: DK_SUB, maxWidth: 520, lineHeight: 1.6 }}>
                Dein {m.discipline} vom {m.weekday} — automatisch aus der Videoaufnahme ausgewertet.
              </p>
              <div style={{ marginTop: 26, display: 'flex', gap: 14, alignItems: 'center' }}>
                <button className="btn-bubble primary small" type="button">Alle Berichte</button>
                <button className="btn-bubble secondary small" type="button">Video ansehen</button>
              </div>
            </div>
            <DeskScoreboard report={report} />
          </div>
        </Band>
      </Reveal>

      <Reveal i={1} anim={anim}>
        <Band bg="#FFFFFF" pad="42px 5% 56px">
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div style={{ width: 300 }}>
              <PlayerTabs theme={THEMES.A} players={report.players} tab={tab} setTab={setTab} />
            </div>
          </div>
          <div style={{ marginTop: 44, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
            {dkItems(s).map((it, i) => (
              <div key={it.label} style={{ padding: '4px 30px', borderLeft: i ? DK_HAIR : 'none' }}>
                <div style={{ fontSize: 11.5, letterSpacing: '0.14em', textTransform: 'uppercase', color: DK_SUB, fontWeight: 600 }}>{it.label}</div>
                <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 54, lineHeight: 1.1, letterSpacing: '-0.02em', marginTop: 6 }}>
                  <CountUp value={it.value} decimals={it.dec} anim={anim} />
                  <span style={{ fontSize: 17, fontWeight: 600, color: DK_SUB, marginLeft: 6 }}>{it.unit}</span>
                </div>
              </div>
            ))}
          </div>
        </Band>
      </Reveal>

      {report.video ? (
        <Reveal i={2} anim={anim}>
          <Band bg="#FEF4EA">
            <SectionHead eyebrow="Video" title="Das Spiel im Video"
              sub="Tracking, Laufwege und Zonen direkt auf der Aufnahme — schalte die Overlays nach Bedarf zu." center />
            <DeskPortalVideo video={report.video} players={report.players} posId={'portal-' + m.id} />
          </Band>
        </Reveal>
      ) : null}

      <Reveal i={3} anim={anim}>
        <Band bg="#FFFFFF">
          <SectionHead eyebrow="Feldabdeckung" title={p.isViewer ? 'Wo du gestanden hast' : 'Wo ' + p.first + ' gestanden hat'} />
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 80, alignItems: 'center', marginTop: 36 }}>
            <div>
              <CourtHeatmap spots={s.heat} style={heat} width={340} anim={anim} />
              <HeatLegend theme={THEMES.A} />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: 18, fontFamily: 'var(--font-heading)', fontWeight: 700 }}>Spielfeld-Zonen</h3>
              <p style={{ margin: '6px 0 18px', fontSize: 13.5, color: DK_SUB }}>Anteil aktiver Spielzeit je Zone.</p>
              <ZoneBlock zones={s.zones} theme={THEMES.A} anim={anim} />
            </div>
          </div>
        </Band>
      </Reveal>

      <Reveal i={4} anim={anim}>
        <Band bg="#FEF4EA">
          <SectionHead eyebrow="Verlauf" title="Aktivität und Schläge" />
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20, marginTop: 32 }}>
            <DeskCard title="Aktivität" sub="Distanz je 5 Minuten">
              <TimelineChart points={s.timeline} theme={THEMES.A} anim={anim} />
            </DeskCard>
            <DeskCard title="Schlagverteilung" sub={deNum(s.shots.reduce((a, x) => a + x[1], 0)) + ' Schläge'}
              foot={s.fastest ? 'Schnellster Smash ≈ ' + deNum(s.fastest) + ' km/h.' : null}>
              <ShotMix shots={s.shots} theme={THEMES.A} anim={anim} />
            </DeskCard>
          </div>
        </Band>
      </Reveal>

      <Reveal i={5} anim={anim}>
        <Band bg="#FFFFFF" pad="56px 5% 64px">
          <SectionHead eyebrow="Ballwechsel" title={deNum(report.rallies.count) + ' Ballwechsel'} center
            sub={'Längster Ballwechsel: ' + report.rallies.longShots + ' Schläge in ' + report.rallies.longSec + '\u2009s. ' + report.rallies.endLine} />
          <div style={{ maxWidth: 560, margin: '34px auto 0' }}>
            <DeskRallies rallies={report.rallies} anim={anim} />
          </div>
        </Band>
      </Reveal>

      <footer style={{ background: '#FEF4EA', borderTop: DK_HAIR, padding: '36px 5%', textAlign: 'center' }}>
        <img src="assets/logo-mark.svg" alt="BC Rheinbach" style={{ height: 30, opacity: 0.9 }} />
        <p style={{ margin: '10px 0 0', fontSize: 12, color: DK_SUB, lineHeight: 1.6 }}>
          Automatisch aus der Videoaufnahme ausgewertet · Nur für Mitglieder sichtbar.<br />
          Du möchtest deine Daten löschen? Schreib uns kurz.
        </p>
      </footer>
    </div>
  );
}

// Portal: zentrierter Video-Player ohne Karte
function DeskPortalVideo({ video, players, posId }) {
  const vp = useDeskVideo(posId);
  return (
    <div style={{ maxWidth: 940, margin: '34px auto 0' }}>
      <DeskVideoSurface vp={vp} video={video} players={players} radius={16} />
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
        <DeskOverlayChips vp={vp} />
      </div>
      <DeskVideoControls vp={vp} style={{ marginTop: 14 }} />
      <p style={{ margin: '12px 0 0', fontSize: 11.5, color: DK_SUB, textAlign: 'center' }}>
        Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage.
      </p>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// 3 · ANALYSE-STUDIO — Video zuerst, Kennzahlen als Seitenleiste
// ───────────────────────────────────────────────────────────────
function DeskStudioScreen({ anim, heat, report }) {
  const [tab, setTab] = React.useState(0);
  const vp = useDeskVideo('studio-' + report.meta.id);
  const p = report.players[tab], s = p.stats, m = report.meta;
  const opponent = report.pairs ? report.pairs[1].label : (report.players[1] ? report.players[1].name : '');

  return (
    <div data-screen-label="Desktop · Analyse-Studio" style={{
      width: 1440, minHeight: '100%', background: '#FEF4EA', color: DK_INK,
      fontFamily: 'var(--font-body)', boxSizing: 'border-box'
    }}>
      <header style={{ background: '#FFFFFF', borderBottom: DK_HAIR, padding: '13px 28px', display: 'flex', alignItems: 'center', gap: 14 }}>
        <GhostBtn theme={THEMES.A} icon={ICONS.arrowL} label="Zurück zur Liste" />
        <img src="assets/logo-mark.svg" style={{ height: 28 }} alt="" />
        <div>
          <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 16, lineHeight: 1.15 }}>
            {m.discipline} gegen {opponent}
          </div>
          <div style={{ fontSize: 12, color: DK_SUB }}>{m.weekday}, {m.dateLabel} · Feld {m.court} · {m.score}</div>
        </div>
        <ResultChip result={m.result} line={m.resultLine} theme={THEMES.A} />
        <div style={{ marginLeft: 'auto', width: 260, flex: 'none' }}>
          <PlayerTabs theme={THEMES.A} players={report.players} tab={tab} setTab={setTab} />
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 396px', gap: 18, padding: '20px 28px 28px', alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 13 }}>
          <Reveal i={0} anim={anim}>
            {report.video ? <DeskVideoSurface vp={vp} video={report.video} players={report.players} radius={16} /> : null}
          </Reveal>
          <Reveal i={1} anim={anim}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <DeskOverlayChips vp={vp} />
              <span style={{ marginLeft: 'auto', fontSize: 12, color: DK_SUB }}>{report.video ? report.video.label : ''} — Demo-Daten auf Beispiel-Footage.</span>
            </div>
          </Reveal>
          <Reveal i={2} anim={anim}>
            <DeskVideoControls vp={vp} />
          </Reveal>
          <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 1fr', gap: 13, marginTop: 5 }}>
            <Reveal i={3} anim={anim} style={{ display: 'grid' }}>
              <DeskCard title="Aktivität" sub="Distanz je 5 Minuten" style={{ height: '100%' }}>
                <TimelineChart points={s.timeline} theme={THEMES.A} anim={anim} />
              </DeskCard>
            </Reveal>
            <Reveal i={4} anim={anim} style={{ display: 'grid' }}>
              <DeskCard title="Ballwechsel" style={{ height: '100%' }}>
                <DeskRallies rallies={report.rallies} anim={anim} />
              </DeskCard>
            </Reveal>
          </div>
        </div>

        <aside style={{ display: 'grid', gap: 13 }}>
          <Reveal i={1} anim={anim}>
            <DeskCard title="Kennzahlen" sub="aktive Spielzeit" dense>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                {dkItems(s).map((it) => <DeskKpiBare key={it.label} it={it} anim={anim} />)}
              </div>
            </DeskCard>
          </Reveal>
          <Reveal i={2} anim={anim}>
            <DeskCard title="Feldabdeckung" sub={p.isViewer ? 'wo du gestanden hast' : 'wo ' + p.first + ' gestanden hat'} dense>
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <CourtHeatmap spots={s.heat} style={heat} width={228} anim={anim} />
              </div>
              <HeatLegend theme={THEMES.A} />
            </DeskCard>
          </Reveal>
          <Reveal i={3} anim={anim}>
            <DeskCard title="Spielfeld-Zonen" sub="Anteil aktiver Spielzeit" dense>
              <ZoneBlock zones={s.zones} theme={THEMES.A} anim={anim} />
            </DeskCard>
          </Reveal>
          <Reveal i={4} anim={anim}>
            <DeskCard title="Schlagverteilung" sub={deNum(s.shots.reduce((a, x) => a + x[1], 0)) + ' Schläge'} dense>
              <ShotMix shots={s.shots} theme={THEMES.A} anim={anim} />
            </DeskCard>
          </Reveal>
        </aside>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// 4 · DASHBOARD — alles auf einem Screen (1440 × 900, kein Scrollen)
// ───────────────────────────────────────────────────────────────
function DeskDashboardScreen({ anim, heat, report }) {
  const [tab, setTab] = React.useState(0);
  const vp = useDeskVideo('dash-' + report.meta.id);
  const p = report.players[tab], s = p.stats, m = report.meta;
  const match = window.BCA.matches.find((x) => x.id === m.id);

  return (
    <div data-screen-label="Desktop · Dashboard" style={{
      width: 1440, height: 900, background: '#FEF4EA', color: DK_INK,
      fontFamily: 'var(--font-body)', boxSizing: 'border-box',
      display: 'flex', flexDirection: 'column', overflow: 'hidden'
    }}>
      <header style={{ height: 62, flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '0 20px', borderBottom: DK_HAIR }}>
        <img src="assets/logo-mark.svg" style={{ height: 30 }} alt="" />
        <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 17 }}>Spielanalyse</span>
        <div className="bca-row-flat" style={{
          display: 'flex', alignItems: 'center', gap: 8, background: '#FFFFFF', border: DK_HAIR,
          borderRadius: 10, padding: '7px 12px', boxShadow: 'var(--shadow-sm)', cursor: 'pointer', marginLeft: 12
        }}>
          <span style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap' }}>{m.dd} · {m.discipline} gegen {match ? match.b : ''}</span>
          <Icon d={DK_ICONS.chevD} size={14} color={DK_SUB} />
        </div>
        <ResultChip result={m.result} line={m.resultLine} theme={THEMES.A} small />
        <span style={{ fontSize: 12.5, color: DK_SUB, fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>
        <div style={{ marginLeft: 'auto', width: 240, flex: 'none' }}>
          <PlayerTabs theme={THEMES.A} players={report.players} tab={tab} setTab={setTab} />
        </div>
      </header>

      <div style={{ flex: 1, minHeight: 0, padding: 16, display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 14, boxSizing: 'border-box' }}>
        {/* Spalte 1 — Video */}
        <div style={{ gridColumn: 'span 5', display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
          <DeskCard title="Video-Analyse" sub={report.video ? report.video.label : ''} dense style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            {report.video ? (
              <React.Fragment>
                <DeskVideoSurface vp={vp} video={report.video} players={report.players} radius={10} />
                <div style={{ marginTop: 10 }}><DeskOverlayChips vp={vp} /></div>
                <DeskVideoControls vp={vp} style={{ marginTop: 10 }} />
              </React.Fragment>
            ) : null}
            <div style={{ marginTop: 'auto', paddingTop: 14 }}>
              <div style={{ borderTop: DK_HAIR, paddingTop: 12 }}>
                <DeskRallies rallies={report.rallies} anim={anim} />
              </div>
            </div>
          </DeskCard>
        </div>

        {/* Spalte 2 — Kennzahlen, Zonen, Schläge */}
        <div style={{ gridColumn: 'span 4', display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
          <DeskCard title="Kennzahlen" sub="aktive Spielzeit" dense>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 13 }}>
              {dkItems(s).map((it) => <DeskKpiBare key={it.label} it={it} anim={anim} />)}
            </div>
          </DeskCard>
          <DeskCard title="Spielfeld-Zonen" sub="Anteil aktiver Spielzeit" dense>
            <ZoneBlock zones={s.zones} theme={THEMES.A} anim={anim} />
          </DeskCard>
          <DeskCard title="Schlagverteilung" sub={deNum(s.shots.reduce((a, x) => a + x[1], 0)) + ' Schläge'} dense style={{ flex: 1 }}>
            <ShotMix shots={s.shots} theme={THEMES.A} anim={anim} />
          </DeskCard>
        </div>

        {/* Spalte 3 — Endstand, Feldabdeckung, Aktivität */}
        <div style={{ gridColumn: 'span 3', display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
          <DeskScoreboard report={report} dense />
          <DeskCard title="Feldabdeckung" sub={p.isViewer ? 'du' : p.first} dense style={{ flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <CourtHeatmap spots={s.heat} style={heat} width={206} anim={anim} />
            </div>
            <HeatLegend theme={THEMES.A} />
          </DeskCard>
          <DeskCard title="Aktivität" sub="je 5 Minuten" dense>
            <TimelineChart points={s.timeline} theme={THEMES.A} anim={anim} />
          </DeskCard>
        </div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// 5 · VIDEO-DASHBOARD — ein Screen, Video dominiert (Option 4 + 3)
// ───────────────────────────────────────────────────────────────
function DeskVideoDashScreen({ anim, heat, report }) {
  const [tab, setTab] = React.useState(0);
  const vp = useDeskVideo('vdash-' + report.meta.id);
  const p = report.players[tab], s = p.stats, m = report.meta;
  const match = window.BCA.matches.find((x) => x.id === m.id);
  const r = report.rallies;

  return (
    <div data-screen-label="Desktop · Video-Dashboard" style={{
      width: 1440, height: 900, background: '#FEF4EA', color: DK_INK,
      fontFamily: 'var(--font-body)', boxSizing: 'border-box',
      display: 'flex', flexDirection: 'column', overflow: 'hidden'
    }}>
      <header style={{ height: 62, flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '0 20px', borderBottom: DK_HAIR }}>
        <img src="assets/logo-mark.svg" style={{ height: 30 }} alt="" />
        <div className="bca-row-flat" style={{
          display: 'flex', alignItems: 'center', gap: 8, background: '#FFFFFF', border: DK_HAIR,
          borderRadius: 10, padding: '7px 12px', boxShadow: 'var(--shadow-sm)', cursor: 'pointer', marginLeft: 12
        }}>
          <span style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap' }}>{m.dd} · {m.discipline} gegen {match ? match.b : ''}</span>
          <Icon d={DK_ICONS.chevD} size={14} color={DK_SUB} />
        </div>
        <div style={{ marginLeft: 'auto', width: 240, flex: 'none' }}>
          <PlayerTabs theme={THEMES.A} players={report.players} tab={tab} setTab={setTab} />
        </div>
      </header>

      <div style={{
        flex: 1, minHeight: 0, padding: 16, boxSizing: 'border-box',
        display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gridTemplateRows: 'minmax(0, 1fr) auto', gap: 14
      }}>
        {/* Video dominiert — ohne Karten-Chrome, wie im Studio */}
        <div style={{ gridColumn: 'span 8', gridRow: '1', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {report.video ? (
            <SafariFrame url="bc-rheinbach.de/spielanalyse/2026-06-10" barH={34}
              style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <DeskVideoSurface vp={vp} video={report.video} players={report.players} fill />
              <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '6px 12px', background: '#FFFFFF', borderTop: DK_HAIR }}>
                <DeskVideoControls vp={vp} style={{ flex: 1 }} />
                <DeskOverlayChips vp={vp} />
              </div>
            </SafariFrame>
          ) : null}
        </div>

        {/* Rechts: Endstand + große Heatmap */}
        <div style={{ gridColumn: 'span 4', gridRow: '1', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <DeskScoreboard report={report} dense
            extra={deNum(r.count) + ' Ballwechsel · Ø ' + deNum(r.avgShots, 1) + ' Schläge · längster ' + r.longShots + ' in ' + r.longSec + '\u2009s'} />
          <DeskCard title="Feldabdeckung" sub={p.isViewer ? 'wo du gestanden hast' : 'wo ' + p.first + ' gestanden hat'} dense
            style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CourtHeatmap spots={s.heat} style={heat} width={200} anim={anim} />
            </div>
            <HeatLegend theme={THEMES.A} />
          </DeskCard>
        </div>

        {/* Untere Reihe: kompakte Kacheln */}
        <DeskCard title="Kennzahlen" sub="aktive Spielzeit" dense style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 13 }}>
            {dkItems(s).map((it) => <DeskKpiBare key={it.label} it={it} anim={anim} />)}
          </div>
        </DeskCard>
        <DeskCard title="Spielfeld-Zonen" sub="aktive Spielzeit" dense style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <ZoneBlock zones={s.zones} theme={THEMES.A} anim={anim} />
        </DeskCard>
        <DeskCard title="Aktivität" sub="je 5 Minuten" dense style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <div style={{ maxWidth: 320, margin: '0 auto' }}>
            <TimelineChart points={s.timeline} theme={THEMES.A} anim={anim} />
          </div>
        </DeskCard>
        <DeskCard title="Schlagverteilung" sub={deNum(s.shots.reduce((a, x) => a + x[1], 0)) + ' Schläge'} dense style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <ShotMix shots={s.shots} theme={THEMES.A} anim={anim} compact />
        </DeskCard>
      </div>
    </div>
  );
}

Object.assign(window, { DeskSplitScreen, DeskPortalScreen, DeskPortalVideo, DeskStudioScreen, DeskDashboardScreen, DeskVideoDashScreen });
