import type { ReactNode } from 'react'
import type { ReportPlayer } from '../../types'
import type { Anim } from '../../lib/motion'
import { Icon, ICONS } from './Icon'
import { CountUp } from './CountUp'

interface Item {
  label: string
  value: number
  dec: number
  unit: string
  icon: ReactNode
}

interface Props {
  player: ReportPlayer
  /** 'cards' = mobile 2×2 mini-cards; 'bare' = desktop tile (no inner cards, smaller). */
  variant?: 'cards' | 'bare'
  anim?: Anim
}

/** Key numbers: Distanz / Sprints / Ø Tempo / Top-Tempo (FR-12). */
export function KeyStats({ player, variant = 'cards', anim = 'dezent' }: Props) {
  const s = player.stats
  const items: Item[] = [
    { label: 'Distanz', value: s.distance_m / 1000, dec: 2, unit: 'km', icon: ICONS.route },
    { label: 'Sprints', value: s.sprints, dec: 0, unit: '', icon: ICONS.bolt },
    { label: 'Ø Tempo', value: s.avg_kmh, dec: 1, unit: 'km/h', icon: ICONS.gauge },
    { label: 'Top-Tempo', value: s.top_kmh, dec: 1, unit: 'km/h', icon: ICONS.clock },
  ]
  const bare = variant === 'bare'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: bare ? 13 : 10 }}>
      {items.map((it) => {
        const inner = (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: bare ? 7 : 8 }}>
              <span
                style={{
                  width: bare ? 24 : 28,
                  height: bare ? 24 : 28,
                  borderRadius: bare ? 8 : 9,
                  background: 'rgba(250,136,50,.13)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#C66C27',
                  flex: 'none',
                }}
              >
                <Icon d={it.icon} size={bare ? 13 : 15} />
              </span>
              <span style={{ fontSize: 11.5, color: 'var(--muted)', fontWeight: 500, whiteSpace: 'nowrap' }}>{it.label}</span>
            </div>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: bare ? 22 : 24, marginTop: bare ? 6 : 8, letterSpacing: '-0.01em' }}>
              <CountUp value={it.value} decimals={it.dec} anim={anim} />
              {it.unit && <span style={{ fontSize: bare ? 12 : 12.5, fontWeight: 600, color: 'var(--muted)', marginLeft: 4 }}>{it.unit}</span>}
            </div>
          </>
        )
        return bare ? (
          <div key={it.label}>{inner}</div>
        ) : (
          <div
            key={it.label}
            style={{
              background: 'var(--card)',
              border: '1px solid var(--hairline)',
              borderRadius: 12,
              boxShadow: 'var(--shadow-sm)',
              padding: '13px 14px',
            }}
          >
            {inner}
          </div>
        )
      })}
    </div>
  )
}
