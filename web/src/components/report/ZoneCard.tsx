import type { CSSProperties } from 'react'
import type { Zones } from '../../types'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'

interface Props {
  zones: Zones
  sub?: string
  dense?: boolean
  style?: CSSProperties
  anim?: Anim
}

/** "Spielfeld-Zonen" — front/mid/rear depth bars + a left/right split. */
export function ZoneCard({ zones, sub = 'Anteil aktiver Spielzeit', dense, style, anim = 'dezent' }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.3)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  const bands = [
    { label: 'Vorderfeld', v: zones.front },
    { label: 'Mittelfeld', v: zones.mid },
    { label: 'Hinterfeld', v: zones.rear },
  ]
  const maxV = Math.max(zones.front, zones.mid, zones.rear) || 1

  return (
    <SectionCard title="Spielfeld-Zonen" sub={sub} dense={dense} style={style}>
      <div ref={ref}>
        <div style={{ display: 'grid', gap: 10 }}>
          {bands.map((b, i) => (
            <div key={b.label} style={{ display: 'grid', gridTemplateColumns: '86px 1fr 44px', gap: 10, alignItems: 'center' }}>
              <span style={{ fontSize: 12.5, color: 'var(--muted)' }}>{b.label}</span>
              <div style={{ height: 12, borderRadius: 6, background: 'rgba(250,136,50,.12)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    borderRadius: 6,
                    width: `${(b.v / maxV) * 100}%`,
                    background: b.v === maxV ? 'var(--orange)' : 'rgba(250,136,50,.42)',
                    transform: on ? 'scaleX(1)' : 'scaleX(0)',
                    transformOrigin: 'left',
                    transition: `transform .7s cubic-bezier(.22,.8,.3,1) ${i * 110}ms`,
                  }}
                />
              </div>
              <strong style={{ fontSize: 14, textAlign: 'right', fontFamily: 'var(--font-heading)', fontVariantNumeric: 'tabular-nums' }}>
                {b.v}&#8201;%
              </strong>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>
            <span>Links {zones.left}&#8201;%</span>
            <span>Rechts {zones.right}&#8201;%</span>
          </div>
          <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
            <div
              style={{
                width: `${zones.left}%`,
                background: 'var(--orange)',
                borderRadius: '6px 0 0 6px',
                transform: on ? 'scaleX(1)' : 'scaleX(0)',
                transformOrigin: 'left',
                transition: 'transform .7s cubic-bezier(.22,.8,.3,1) .25s',
              }}
            />
            <div
              style={{
                width: `${zones.right}%`,
                background: 'rgba(250,136,50,.12)',
                borderRadius: '0 6px 6px 0',
                transform: on ? 'scaleX(1)' : 'scaleX(0)',
                transformOrigin: 'right',
                transition: 'transform .7s cubic-bezier(.22,.8,.3,1) .25s',
              }}
            />
          </div>
        </div>
      </div>
    </SectionCard>
  )
}
