import { useState } from 'react'
import { useParams, useSearchParams, Link } from 'react-router-dom'
import { useMatchReport } from '../hooks/useMatchReport'
import { useMediaQuery } from '../hooks/useMediaQuery'
import { KeyStats } from '../components/report/KeyStats'
import { SectionCard } from '../components/report/SectionCard'
import { disciplineDE, shotDE } from '../lib/labels'

// P0 SHELL: renders meta + player tabs + KPIs from real data. Full viz (heatmap, zones, timeline,
// shot mix, video overlays) and the pixel-perfect mobile/desktop layouts land in P1 (specs/50-web-app.md).
export function MatchReport() {
  const { slug } = useParams()
  const [params] = useSearchParams()
  const viewerPlayerId = params.get('p') ?? undefined
  const { loading, error, report } = useMatchReport(slug, viewerPlayerId)
  const isDesktop = useMediaQuery('(min-width: 1200px)')
  const [tab, setTab] = useState(0)

  if (loading) return <Centered>Lädt…</Centered>
  if (error || !report) return <Centered>{error ?? 'Kein Bericht gefunden.'}</Centered>

  const player = report.players[tab] ?? report.players[0]
  const m = report.meta

  return (
    <main style={{ maxWidth: isDesktop ? 1200 : 480, margin: '0 auto', padding: '8px 14px 28px' }}>
      <Link to="/" style={{ fontSize: 13, color: 'var(--muted)' }}>‹ Zurück</Link>

      <header style={{ padding: '8px 4px 12px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--orange)' }}>Spielbericht</div>
        <h2 style={{ fontSize: 27, fontWeight: 700 }}>
          {m.result === 'win' ? 'Stark gespielt' : 'Gut gekämpft'}, {player.first}.
        </h2>
        <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 2 }}>
          {m.weekday}, {m.dateLabel} · {disciplineDE(m.discipline)}{m.court ? ` · Feld ${m.court}` : ''}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8 }}>
          <span style={{ fontSize: 11, fontWeight: 700, borderRadius: 99, padding: '2px 10px', background: m.result === 'win' ? 'rgba(250,136,50,.16)' : 'rgba(10,10,11,.07)', color: m.result === 'win' ? 'var(--orange-shade)' : 'var(--muted)' }}>{m.resultLine}</span>
          <span className="tabular" style={{ fontSize: 13, color: 'var(--muted)' }}>{m.score}</span>
        </div>
      </header>

      {/* player tabs */}
      <div style={{ display: 'flex', gap: 4, background: 'rgba(10,10,11,.05)', border: '1px solid var(--hairline)', borderRadius: 12, padding: 4, marginBottom: 12 }}>
        {report.players.map((p, i) => (
          <button
            key={p.name + i}
            onClick={() => setTab(i)}
            style={{
              flex: 1, borderRadius: 9, padding: '8px 4px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              border: 'none', fontFamily: 'var(--font-body)',
              background: i === tab ? 'var(--card)' : 'transparent',
              boxShadow: i === tab ? 'var(--shadow-sm)' : 'none',
              color: i === tab ? 'var(--ink)' : 'var(--muted)',
            }}
          >
            {p.isViewer ? 'Du' : p.first}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <KeyStats player={player} />

        <SectionCard title="Schlagverteilung" sub={`${player.stats.shots.reduce((n, s) => n + s[1], 0)} Schläge`}>
          {player.stats.shots.length === 0 ? (
            <p style={{ color: 'var(--muted)', fontSize: 13 }}>Noch keine Schlagdaten.</p>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: 13 }}>
              {player.stats.shots.map(([name, count]) => (
                <li key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                  <span style={{ color: 'var(--muted)' }}>{shotDE(name)}</span>
                  <span style={{ fontFamily: 'var(--font-head)', fontWeight: 700 }}>{count}</span>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>

        {report.rallies && (
          <SectionCard title="Ballwechsel">
            <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
              <Stat label="gesamt" value={report.rallies.count} />
              <Stat label="Ø Schläge" value={report.rallies.avgShots} />
              <Stat label="längster" value={report.rallies.longShots} />
            </div>
            {report.rallies.endLine && (
              <p style={{ textAlign: 'center', fontSize: 12.5, color: 'var(--muted)', marginTop: 10 }}>{report.rallies.endLine}</p>
            )}
          </SectionCard>
        )}

        <p style={{ fontSize: 11.5, color: 'var(--muted)', textAlign: 'center' }}>
          Layout: {isDesktop ? 'Desktop-Dashboard' : 'Mobil'} · vollständige Visualisierung folgt in P1.
        </p>
      </div>
    </main>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 22 }}>{value}</div>
      <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>{label}</div>
    </div>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
}
