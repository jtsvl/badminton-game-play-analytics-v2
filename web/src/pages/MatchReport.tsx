import { useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useMatchReport } from '../hooks/useMatchReport'
import { useMediaQuery } from '../hooks/useMediaQuery'
import { MatchReportMobile } from './MatchReportMobile'
import { MatchReportDashboard } from './MatchReportDashboard'

// One route, one fetch. The viewport picks the layout; `tab` is lifted here so it survives a
// remount when the breakpoint is crossed. See specs/50-web-app.md + specs/features/frontend-port.md.
export function MatchReport() {
  const { slug } = useParams()
  const [params] = useSearchParams()
  const viewerPlayerId = params.get('p') ?? undefined
  const { loading, error, report } = useMatchReport(slug, viewerPlayerId)
  const isDesktop = useMediaQuery('(min-width: 1200px)')
  const [tab, setTab] = useState(0)

  if (loading) return <Centered>Lädt…</Centered>
  if (error || !report) return <Centered>{error ?? 'Kein Bericht gefunden.'}</Centered>

  const safeTab = tab < report.players.length ? tab : 0
  return isDesktop ? (
    <MatchReportDashboard report={report} tab={safeTab} setTab={setTab} slug={slug} />
  ) : (
    <MatchReportMobile report={report} tab={safeTab} setTab={setTab} slug={slug} />
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
}
