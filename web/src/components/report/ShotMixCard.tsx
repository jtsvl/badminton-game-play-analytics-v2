import type { CSSProperties } from 'react'
import type { ShotPair } from '../../types'
import { deNum } from '../../lib/format'
import { shotDE } from '../../lib/labels'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'

interface Props {
  shots: ShotPair[]
  fastest?: number | null
  /** „Aus-Schläge" aus Rally-Ende-Landungen (Schätzung, Untergrenze); null/undefined = ausblenden. */
  outShots?: number | null
  dense?: boolean
  /** Tighter rows + no per-row percentage, for the narrow desktop tile. */
  compact?: boolean
  style?: CSSProperties
  anim?: Anim
}

/** "Schlagverteilung" — horizontal bars per shot type (canonical → German via shotDE), sorted desc. */
export function ShotMixCard({ shots, fastest, outShots, dense, compact, style, anim = 'dezent' }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.2)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  const total = shots.reduce((a, s) => a + s[1], 0)
  const sub = `${deNum(total)} Schläge`
  const outLine = outShots != null
    ? `Aus-Schläge: ${deNum(outShots)} (Schätzung, Untergrenze – nur sicher erkannte Landungen).`
    : undefined
  const fastLine = fastest ? `Schnellster Smash ≈ ${deNum(fastest)} km/h — Schätzung aus der Shuttle-Verfolgung.` : undefined
  const foot = [fastLine, outLine].filter(Boolean).join(' ') || undefined

  if (shots.length === 0) {
    // Bis Stage 7 (stroke) existiert, haben echte Spiele keine Schlagtypen, aber ggf. Aus-Schläge —
    // der foot muss auch im Leerzustand erscheinen.
    return (
      <SectionCard title="Schlagverteilung" sub={sub} foot={foot} dense={dense} style={style}>
        <p style={{ fontSize: 13, color: 'var(--muted)', margin: 0 }}>Noch keine Schlagdaten.</p>
      </SectionCard>
    )
  }

  const sorted = [...shots].sort((a, b) => b[1] - a[1])
  const max = sorted[0][1] || 1
  const cols = compact ? '70px 1fr 36px' : '76px 1fr 58px'

  return (
    <SectionCard title="Schlagverteilung" sub={sub} foot={foot} dense={dense} style={style}>
      <div ref={ref} style={{ display: 'grid', gap: compact ? 6.5 : 9 }}>
        {sorted.map((s, i) => {
          const pct = Math.round((s[1] / total) * 100)
          const top = i === 0
          return (
            <div key={s[0]} style={{ display: 'grid', gridTemplateColumns: cols, gap: 10, alignItems: 'center' }}>
              <span style={{ fontSize: compact ? 12 : 12.5, color: top ? 'var(--ink)' : 'var(--muted)', fontWeight: top ? 600 : 400 }}>
                {shotDE(s[0])}
              </span>
              <div style={{ height: compact ? 9 : 10, borderRadius: 5, background: 'rgba(250,136,50,.12)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    borderRadius: 5,
                    width: `${(s[1] / max) * 100}%`,
                    background: top ? '#FA8832' : 'rgba(250,136,50,.42)',
                    transform: on ? 'scaleX(1)' : 'scaleX(0)',
                    transformOrigin: 'left',
                    transition: `transform .6s cubic-bezier(.22,.8,.3,1) ${i * 70}ms`,
                  }}
                />
              </div>
              <span style={{ fontSize: compact ? 12 : 13, textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontFamily: 'var(--font-heading)', fontWeight: 700 }}>
                {s[1]}
                {!compact && <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 11 }}> · {pct}%</span>}
              </span>
            </div>
          )
        })}
      </div>
    </SectionCard>
  )
}
