import type { CSSProperties, ReactNode } from 'react'
import type { MatchReport, Side } from '../../types'
import { disciplineDE } from '../../lib/labels'
import { ResultChip } from './ResultChip'

interface Props {
  report: MatchReport
  dense?: boolean
  /** Extra line below the foot row (e.g. the rally summary on the dashboard). */
  extra?: ReactNode
  style?: CSSProperties
}

/** "Endstand" scoreboard card — per-game score cells (won = orange), result + duration, optional extra line. */
export function Endstand({ report, dense, extra, style }: Props) {
  const m = report.meta
  const sideName = (side: Side): string => {
    if (report.pairs && report.pairs.length) {
      return report.pairs.find((x) => x.side === side)?.label ?? ''
    }
    const pl = report.players.find((x) => x.side === side)
    if (!pl) return ''
    return pl.isViewer ? `${pl.name} (du)` : pl.name
  }

  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--hairline)',
        borderRadius: 'var(--radius-scoreboard)',
        boxShadow: dense ? 'var(--shadow-sm)' : 'var(--shadow-md)',
        padding: dense ? '14px 16px 16px' : '20px 22px',
        boxSizing: 'border-box',
        ...style,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
        <span className="eyebrow" style={{ fontSize: 11 }}>Endstand</span>
        <span style={{ fontSize: 12, color: 'var(--muted)', whiteSpace: 'nowrap' }}>
          {disciplineDE(m.discipline)}{m.court ? ` · Feld ${m.court}` : ''}
        </span>
      </div>
      <div style={{ marginTop: dense ? 12 : 16, display: 'grid', gap: 10 }}>
        {(['A', 'B'] as Side[]).map((side) => (
          <div key={side} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span
              style={{
                flex: 1,
                fontFamily: 'var(--font-heading)',
                fontWeight: side === 'A' ? 700 : 500,
                fontSize: dense ? 13.5 : 15.5,
                color: side === 'A' ? 'var(--ink)' : 'var(--muted)',
                minWidth: 0,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {sideName(side)}
            </span>
            <span style={{ display: 'flex', gap: 6, flex: 'none' }}>
              {m.games.map((g, gi) => {
                const v = side === 'A' ? g[0] : g[1]
                const won = side === 'A' ? g[0] > g[1] : g[1] > g[0]
                return (
                  <span
                    key={gi}
                    style={{
                      width: dense ? 28 : 32,
                      height: dense ? 25 : 28,
                      borderRadius: 7,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontFamily: 'var(--font-heading)',
                      fontWeight: 700,
                      fontSize: dense ? 13 : 14,
                      fontVariantNumeric: 'tabular-nums',
                      background: won ? '#FA8832' : 'rgba(10,10,11,.06)',
                      color: won ? '#1A0500' : 'var(--muted)',
                    }}
                  >
                    {v}
                  </span>
                )
              })}
            </span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: dense ? 12 : 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <ResultChip result={m.result} line={m.resultLine} small={dense} />
        <span style={{ fontSize: dense ? 11.5 : 12, color: 'var(--muted)', textAlign: 'right' }}>
          {m.durationMin ?? '–'} Min Spielzeit · {m.restMin ?? '–'} Min Pause
        </span>
      </div>
      {extra && (
        <div style={{ marginTop: 11, paddingTop: 10, borderTop: '1px solid var(--hairline)', fontSize: 11.5, color: 'var(--muted)', lineHeight: 1.5 }}>
          {extra}
        </div>
      )}
    </div>
  )
}
