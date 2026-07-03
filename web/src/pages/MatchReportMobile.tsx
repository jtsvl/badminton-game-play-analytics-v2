import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import type { MatchReport } from '../types'
import { overlayInputs } from '../lib/calibration'
import { Reveal } from '../components/report/Reveal'
import { ReportHero } from '../components/report/ReportHero'
import { PlayerTabs } from '../components/report/PlayerTabs'
import { KeyStats } from '../components/report/KeyStats'
import { VideoAnalysisCard } from '../components/report/VideoAnalysisCard'
import { CourtCoverageCard } from '../components/report/CourtCoverageCard'
import { ZoneCard } from '../components/report/ZoneCard'
import { ActivityCard } from '../components/report/ActivityCard'
import { ShotMixCard } from '../components/report/ShotMixCard'
import { RallyCard } from '../components/report/RallyCard'
import { FormationCard } from '../components/report/FormationCard'

interface Props {
  report: MatchReport
  tab: number
  setTab: (i: number) => void
  slug: string | undefined
}

/** Mobile Spielbericht — stacked column (handovers/design_handoff_match_report_app, theme A). */
export function MatchReportMobile({ report, tab, setTab, slug }: Props) {
  const navigate = useNavigate()
  const [run, setRun] = useState(0)
  const p = report.players[tab] ?? report.players[0]
  const s = p.stats
  const coverageSub = p.isViewer ? 'wo du gestanden hast' : `wo ${p.first} gestanden hat`

  // Sections in column order; conditionals keep the stacked column gap-free.
  const oi = overlayInputs(report)
  const sections: ReactNode[] = []
  sections.push(<KeyStats player={p} />)
  if (report.video) sections.push(<VideoAnalysisCard slug={slug} video={report.video} players={report.players} oi={oi} />)
  sections.push(<CourtCoverageCard spots={s.heat} sub={coverageSub} width={222} />)
  sections.push(<ZoneCard zones={s.zones} />)
  sections.push(<ActivityCard points={s.timeline} />)
  // outShots allein (ohne Schlagtypen, bis Stage 7) rendert die Karte im Leerzustand mit Aus-Schläge-foot
  if (s.shots.length > 0 || s.outShots != null)
    sections.push(<ShotMixCard shots={s.shots} fastest={s.fastest} outShots={s.outShots} />)
  const canReviewRallies = ['review', 'done'].includes(report.meta.status)
  if (report.rallies)
    sections.push(
      <RallyCard rallies={report.rallies} reviewTo={canReviewRallies ? `/match/${slug}/rallies` : undefined} />,
    )
  if (report.pairs && report.pairs.length) sections.push(<FormationCard pairs={report.pairs} />)

  return (
    <main style={{ width: '100%', maxWidth: 430, margin: '0 auto', minHeight: '100%', background: 'var(--beige)', paddingBottom: 28 }}>
      <div key={run}>
        <Reveal i={0}>
          <ReportHero report={report} onBack={() => navigate('/')} onReplay={() => setRun((r) => r + 1)} />
        </Reveal>

        <div style={{ padding: '0 14px', marginTop: 16 }}>
          <Reveal i={1}>
            <PlayerTabs players={report.players} tab={tab} setTab={setTab} />
          </Reveal>
        </div>

        <div key={`tab${tab}`} style={{ padding: '0 14px', marginTop: 14, display: 'grid', gap: 12 }}>
          {sections.map((node, i) => (
            <Reveal key={i} i={i + 2}>
              {node}
            </Reveal>
          ))}
          <Reveal i={sections.length + 2}>
            <footer style={{ textAlign: 'center', padding: '14px 10px 4px' }}>
              <img src="/logo-mark.svg" alt="BC Rheinbach" style={{ height: 26, opacity: 0.85 }} />
              <p style={{ margin: '8px 0 0', fontSize: 11, color: 'var(--muted)', lineHeight: 1.55 }}>
                Automatisch aus der Videoaufnahme ausgewertet.
                <br />
                Du möchtest deine Daten löschen? Schreib uns kurz.
              </p>
            </footer>
          </Reveal>
        </div>
      </div>
    </main>
  )
}
