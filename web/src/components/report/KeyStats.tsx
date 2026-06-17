import { deNum, deKm } from '../../lib/format'
import type { ReportPlayer } from '../../types'

interface Props { player: ReportPlayer }

/** 2×2 KPI grid: Distanz / Sprints / Ø Tempo / Top-Tempo (specs FR-12). */
export function KeyStats({ player }: Props) {
  const s = player.stats
  const cells: Array<{ label: string; value: string; unit?: string }> = [
    { label: 'Distanz', value: deKm(s.distance_m), unit: 'km' },
    { label: 'Sprints', value: deNum(s.sprints) },
    { label: 'Ø Tempo', value: deNum(s.avg_kmh, 1), unit: 'km/h' },
    { label: 'Top-Tempo', value: deNum(s.top_kmh, 1), unit: 'km/h' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      {cells.map((c) => (
        <div key={c.label} style={{ background: 'var(--card)', border: '1px solid var(--hairline)', borderRadius: 12, boxShadow: 'var(--shadow-sm)', padding: '13px 14px' }}>
          <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>{c.label}</div>
          <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 24 }}>
            {c.value}
            {c.unit && <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--muted)' }}> {c.unit}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}
