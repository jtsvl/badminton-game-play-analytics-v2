import type { CSSProperties } from 'react'
import type { PairStat } from '../../types'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'

interface Props {
  pairs: PairStat[]
  dense?: boolean
  style?: CSSProperties
  anim?: Anim
}

/** "Doppel-Formation" — front-back vs side-by-side split per pair (doubles only). */
export function FormationCard({ pairs, dense, style, anim = 'dezent' }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.3)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  return (
    <SectionCard title="Doppel-Formation" sub="Anteil aktiver Spielzeit" dense={dense} style={style}>
      <div ref={ref} style={{ display: 'grid', gap: 18 }}>
        {pairs.map((pair, pi) => (
          <div key={pair.side}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 7 }}>
              <strong style={{ fontSize: 13.5, fontFamily: 'var(--font-heading)' }}>{pair.label}</strong>
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>{pair.rot} Rotationen</span>
            </div>
            <div style={{ display: 'flex', height: 14, borderRadius: 7, overflow: 'hidden', gap: 2 }}>
              <div
                style={{
                  width: `${pair.fb}%`,
                  background: '#FA8832',
                  borderRadius: '7px 0 0 7px',
                  transform: on ? 'scaleX(1)' : 'scaleX(0)',
                  transformOrigin: 'left',
                  transition: `transform .7s cubic-bezier(.22,.8,.3,1) ${pi * 140}ms`,
                }}
              />
              <div
                style={{
                  width: `${pair.ss}%`,
                  background: 'rgba(10,10,11,.16)',
                  borderRadius: '0 7px 7px 0',
                  transform: on ? 'scaleX(1)' : 'scaleX(0)',
                  transformOrigin: 'right',
                  transition: `transform .7s cubic-bezier(.22,.8,.3,1) ${pi * 140 + 80}ms`,
                }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--muted)', marginTop: 5 }}>
              <span>Vorne/Hinten {pair.fb}&#8201;%</span>
              <span>Nebeneinander {pair.ss}&#8201;%</span>
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  )
}
