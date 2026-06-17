// app-final.jsx — Finale App: Richtung A (Vereinsfarben), dezente Animation,
// Raster-Heatmap, Video-Analyse mit Overlays.
// Depends on: tweaks-panel.jsx, viz.jsx, screens.jsx, video.jsx, mock-data.js

const FINAL_DEFAULTS = /*EDITMODE-BEGIN*/{
  "anim": "dezent",
  "heat": "raster"
}/*EDITMODE-END*/;

const FINAL_ROUTE_KEY = 'bca-app-route-v1';

function AppPageFade({ children, anim }) {
  const skip = anim === 'aus' || window.BCA_REDUCED;
  const [on, setOn] = React.useState(skip);
  React.useEffect(() => {
    if (on) return;
    const t = setTimeout(() => setOn(true), 30);
    return () => clearTimeout(t);
  }, []);
  return (
    <div style={{
      opacity: on ? 1 : 0, transform: on ? 'none' : 'translateX(26px)',
      transition: 'opacity .3s ease, transform .32s cubic-bezier(.22,.8,.3,1)'
    }}>{children}</div>
  );
}

function FinalApp() {
  const [t, setTweak] = useTweaks(FINAL_DEFAULTS);
  const th = THEMES.A;
  const [route, setRoute] = React.useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(FINAL_ROUTE_KEY));
      if (saved && saved.name === 'report' && window.BCA.reports[saved.id]) return saved;
    } catch (e) { /* noop */ }
    return { name: 'list' };
  });
  const scrollRef = React.useRef(null);

  const nav = (r) => {
    setRoute(r);
    try { localStorage.setItem(FINAL_ROUTE_KEY, JSON.stringify(r)); } catch (e) { /* noop */ }
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  };

  const content = route.name === 'report' ? (
    <ReportScreen theme={th} anim={t.anim} report={window.BCA.reports[route.id]}
      heatOverride={t.heat} onBack={() => nav({ name: 'list' })} />
  ) : (
    <MatchListScreen theme={th} anim={t.anim}
      onOpen={(m) => nav({ name: 'report', id: m.id })} />
  );

  return (
    <div style={{ minHeight: '100dvh', background: '#E9E5DF', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px 0' }}>
      <div className="proto-shell" style={{
        width: 390, height: 'min(844px, calc(100dvh - 48px))', borderRadius: 34, overflow: 'hidden',
        position: 'relative', background: th.pageBg, border: '1px solid rgba(0,0,0,.1)',
        boxShadow: '0 24px 60px rgba(0,0,0,.28)'
      }}>
        <div ref={scrollRef} className="proto-scroll" style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}>
          <AppPageFade key={route.name + '-' + (route.id || 0)} anim={t.anim}>
            {content}
          </AppPageFade>
        </div>
        <div style={{ position: 'absolute', bottom: 6, left: '50%', transform: 'translateX(-50%)', width: 120, height: 5, borderRadius: 3, background: 'rgba(10,10,11,.25)', pointerEvents: 'none' }}></div>
      </div>

      <TweaksPanel>
        <TweakSection label="Animation" />
        <TweakRadio label="Intensität" value={t.anim} options={['aus', 'dezent', 'voll']}
          onChange={(v) => setTweak('anim', v)} />
        <TweakSection label="Heatmap" />
        <TweakSelect label="Stil" value={t.heat} options={['raster', 'blobs', 'punkte']}
          onChange={(v) => setTweak('heat', v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<FinalApp />);
