import { Link } from 'react-router-dom'
import type { MatchReport } from '../types'
import { deNum } from '../lib/format'
import { disciplineDE } from '../lib/labels'
import { overlayInputs } from '../lib/calibration'
import { useVideoPlayer } from '../hooks/useVideoPlayer'
import { Reveal } from '../components/report/Reveal'
import { Icon, ICONS } from '../components/report/Icon'
import { PlayerTabs } from '../components/report/PlayerTabs'
import { SafariFrame } from '../components/report/SafariFrame'
import { VideoOverlay } from '../components/report/VideoOverlay'
import { VideoControls } from '../components/report/VideoControls'
import { OverlayChips } from '../components/report/OverlayChips'
import { Endstand } from '../components/report/Endstand'
import { SectionCard } from '../components/report/SectionCard'
import { CourtCoverageCard } from '../components/report/CourtCoverageCard'
import { KeyStats } from '../components/report/KeyStats'
import { ZoneCard } from '../components/report/ZoneCard'
import { ActivityCard } from '../components/report/ActivityCard'
import { ShotMixCard } from '../components/report/ShotMixCard'

interface Props {
  report: MatchReport
  tab: number
  setTab: (i: number) => void
  slug: string | undefined
}

/** Desktop Video-Dashboard — DeskVideoDashScreen (handovers/design_handoff_desktop_dashboard). */
export function MatchReportDashboard({ report, tab, setTab, slug }: Props) {
  const p = report.players[tab] ?? report.players[0]
  const s = p.stats
  const m = report.meta
  const r = report.rallies
  const hasFootage = Boolean(report.video?.src)
  const vp = useVideoPlayer(slug, hasFootage)
  const oi = overlayInputs(report)

  const opponent =
    report.pairs && report.pairs.length
      ? report.pairs.find((x) => x.side === 'B')?.label ?? ''
      : report.players.find((x) => x.side === 'B')?.name ?? ''
  const coverageSub = p.isViewer ? 'wo du gestanden hast' : `wo ${p.first} gestanden hat`
  const rallyExtra = r
    ? `${deNum(r.count)} Ballwechsel · Ø ${deNum(r.avgShots, 1)} Schläge · längster ${r.longShots} in ${r.longSec} s`
    : undefined

  return (
    <div style={{ minHeight: '100vh', background: 'var(--beige)', color: 'var(--ink)', fontFamily: 'var(--font-body)', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <header style={{ height: 62, flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '0 20px', borderBottom: '1px solid var(--hairline)' }}>
        <img src="/logo-mark.svg" alt="BC Rheinbach" style={{ height: 30 }} />
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            background: 'var(--card)',
            border: '1px solid var(--hairline)',
            borderRadius: 10,
            padding: '7px 12px',
            boxShadow: 'var(--shadow-sm)',
            marginLeft: 12,
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap' }}>
            {m.dd} · {disciplineDE(m.discipline)} gegen {opponent}
          </span>
          <Icon d={ICONS.chevD} size={14} color="var(--muted)" />
        </div>
        <div style={{ marginLeft: 'auto', width: 240, flex: 'none' }}>
          <PlayerTabs players={report.players} tab={tab} setTab={setTab} />
        </div>
      </header>

      {/* Body grid */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          padding: 16,
          boxSizing: 'border-box',
          display: 'grid',
          gridTemplateColumns: 'repeat(12, 1fr)',
          gridTemplateRows: 'minmax(0, 1fr) auto',
          gap: 14,
        }}
      >
        {/* Row 1, left (span 8): video dominates, in a Safari frame */}
        <div style={{ gridColumn: 'span 8', gridRow: '1', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {report.video ? (
            <SafariFrame url={`bc-rheinbach.de/spielanalyse/${slug ?? ''}`} barH={34} style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0A0A0B' }}>
                <div style={{ position: 'relative', height: '100%', maxWidth: '100%', aspectRatio: '16 / 9', lineHeight: 0 }}>
                  {hasFootage && (
                    <video
                      ref={vp.vref}
                      src={report.video.src}
                      muted
                      playsInline
                      preload="metadata"
                      onLoadedMetadata={vp.onLoaded}
                      onEnded={() => vp.setPlaying(false)}
                      onTimeUpdate={() => {
                        const v = vp.vref.current
                        if (v && v.paused) vp.setTime(v.currentTime)
                      }}
                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  )}
                  <VideoOverlay time={vp.time} players={report.players} ov={vp.ov} H={oi.H} tracks={oi.tracks} />
                </div>
              </div>
              <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '6px 12px', background: 'var(--card)', borderTop: '1px solid var(--hairline)' }}>
                <VideoControls player={vp} style={{ flex: 1 }} />
                <OverlayChips ov={vp.ov} toggle={vp.toggle} />
              </div>
            </SafariFrame>
          ) : (
            <SectionCard style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <p style={{ color: 'var(--muted)', fontSize: 14, textAlign: 'center', margin: 0 }}>Für dieses Spiel liegt keine Videoanalyse vor.</p>
            </SectionCard>
          )}
        </div>

        {/* Row 1, right (span 4): Endstand + Feldabdeckung */}
        <div style={{ gridColumn: 'span 4', gridRow: '1', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Endstand report={report} dense extra={rallyExtra} />
          {r && ['review', 'done'].includes(m.status) && (
            <Link
              to={`/match/${slug}/rallies`}
              style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--orange-shade, #C66C27)', alignSelf: 'flex-end', marginTop: -6 }}
            >
              Ballwechsel prüfen →
            </Link>
          )}
          <CourtCoverageCard
            spots={s.heat}
            sub={coverageSub}
            dense
            fill
            width={200}
            style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
          />
        </div>

        {/* Row 2: four dense tiles */}
        <Reveal i={0} style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <SectionCard title="Kennzahlen" sub="aktive Spielzeit" dense>
            <KeyStats player={p} variant="bare" />
          </SectionCard>
        </Reveal>
        <Reveal i={1} style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <ZoneCard zones={s.zones} sub="aktive Spielzeit" dense />
        </Reveal>
        <Reveal i={2} style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <ActivityCard points={s.timeline} sub="je 5 Minuten" chartMaxWidth={320} dense />
        </Reveal>
        <Reveal i={3} style={{ gridColumn: 'span 3', gridRow: '2' }}>
          <ShotMixCard shots={s.shots} fastest={s.fastest} dense compact />
        </Reveal>
      </div>
    </div>
  )
}
