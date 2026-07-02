import type { CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import type { RallySummary } from '../../types'
import { type Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'
import { CountUp } from './CountUp'

interface Props {
  rallies: RallySummary
  dense?: boolean
  style?: CSSProperties
  anim?: Anim
  reviewTo?: string // when set, a "Ballwechsel prüfen" link to the review page (FR-35)
}

/** "Ballwechsel" — gesamt / Ø Schläge / längster, plus a one-line summary. */
export function RallyCard({ rallies, dense, style, anim = 'dezent', reviewTo }: Props) {
  const cells: Array<{ value: number; dec: number; label: string }> = [
    { value: rallies.count, dec: 0, label: 'gesamt' },
    { value: rallies.avgShots, dec: 1, label: 'Ø Schläge' },
    { value: rallies.longShots, dec: 0, label: 'längster' },
  ]
  return (
    <SectionCard title="Ballwechsel" dense={dense} style={style}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, textAlign: 'center' }}>
        {cells.map((c) => (
          <div key={c.label}>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 22 }}>
              <CountUp value={c.value} decimals={c.dec} anim={anim} />
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 2 }}>{c.label}</div>
          </div>
        ))}
      </div>
      <p style={{ margin: '12px 0 0', fontSize: 12.5, color: 'var(--muted)', textAlign: 'center', lineHeight: 1.5 }}>
        Längster Ballwechsel: {rallies.longShots} Schläge in {rallies.longSec}&#8201;s. {rallies.endLine}
      </p>
      {reviewTo && (
        <p style={{ margin: '10px 0 0', textAlign: 'center' }}>
          <Link to={reviewTo} style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--orange-shade, #C66C27)' }}>
            Ballwechsel prüfen →
          </Link>
        </p>
      )}
    </SectionCard>
  )
}
