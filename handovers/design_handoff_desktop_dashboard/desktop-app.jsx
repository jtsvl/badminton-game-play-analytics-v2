// desktop-app.jsx — Design canvas: 4 Desktop-Richtungen (alle Richtung A)
// Depends on: design-canvas.jsx, tweaks-panel.jsx, viz.jsx, screens.jsx,
// video.jsx, desktop-shared.jsx, desktop-screens.jsx, mock-data.js

const DESK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "anim": "dezent",
  "heat": "raster"
}/*EDITMODE-END*/;

// Artboard-Höhen (px) — an den Inhalt angepasst
const DESK_H = { split: 1640, portal: 3380, studio: 1170, dash: 900, vdash: 900 };

const DESK_SECTIONS = [
  {
    id: 'vdash', title: 'Video-Dashboard',
    blurb: 'Ein Screen ohne Scrollen, das Video dominiert: große Bühne im Safari-Rahmen links, Endstand und Heatmap daneben, kompakte Kacheln darunter.',
    Comp: 'DeskVideoDashScreen'
  }
];

function DesktopApp() {
  const [t, setTweak] = useTweaks(DESK_DEFAULTS);
  const report = window.BCA.reports[1]; // Einzel mit Video-Analyse

  // Dev measurement: log real content heights vs card heights
  React.useEffect(() => {
    const id = setTimeout(() => {
      document.querySelectorAll('.dc-card').forEach((card) => {
        const sc = card.firstElementChild;
        if (sc && sc.dataset && sc.dataset.screenLabel) {
          console.log('[measure] ' + sc.dataset.screenLabel + ' content=' + sc.scrollHeight + ' card=' + card.clientHeight);
        }
      });
    }, 1500);
    return () => clearTimeout(id);
  }, []);

  return (
    <React.Fragment>
      <DesignCanvas>
        {DESK_SECTIONS.map((sec) => {
          const Screen = window[sec.Comp];
          return (
            <DCSection key={sec.id} id={'desk-' + sec.id} title={sec.title} subtitle={sec.blurb}>
              <DCArtboard id={'desk-' + sec.id + '-board'} label="Spielbericht · 1440" width={1440} height={DESK_H[sec.id]}>
                <Screen anim={t.anim} heat={t.heat} report={report} />
              </DCArtboard>
            </DCSection>
          );
        })}
      </DesignCanvas>

      <TweaksPanel>
        <TweakSection label="Animation" />
        <TweakRadio label="Intensität" value={t.anim} options={['aus', 'dezent', 'voll']}
          onChange={(v) => setTweak('anim', v)} />
        <TweakSection label="Heatmap" />
        <TweakSelect label="Stil" value={t.heat} options={['raster', 'blobs', 'punkte']}
          onChange={(v) => setTweak('heat', v)} />
        <TweakSection label="Mehr" />
        <TweakButton label="Mobile App öffnen" onClick={() => { window.location.href = 'Spielanalyse App.html'; }} />
      </TweaksPanel>
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<DesktopApp />);
