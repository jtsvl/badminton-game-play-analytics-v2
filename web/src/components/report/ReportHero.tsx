import type { ReactNode } from 'react'
import type { MatchReport } from '../../types'
import { disciplineDE } from '../../lib/labels'
import { Icon, ICONS } from './Icon'
import { ResultChip } from './ResultChip'

function GhostBtn({ icon, label, onClick }: { icon: ReactNode; label: string; onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      style={{
        width: 34,
        height: 34,
        borderRadius: 17,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid var(--hairline)',
        background: 'var(--card)',
        color: 'var(--ink)',
        cursor: 'pointer',
        padding: 0,
        flex: 'none',
      }}
    >
      <Icon d={icon} size={16} />
    </button>
  )
}

interface Props {
  report: MatchReport
  onBack?: () => void
  onReplay?: () => void
}

/** Mobile report hero (theme A): greeting addressed to the viewer + result chip + score. */
export function ReportHero({ report, onBack, onReplay }: Props) {
  const m = report.meta
  const won = m.result === 'win'
  const viewer = report.players.find((p) => p.isViewer) ?? report.players[0]
  const first = viewer?.first ?? ''
  const headline = won ? `Stark gespielt, ${first}.` : `Gut gekämpft, ${first}.`

  return (
    <header style={{ padding: '4px 18px 0' }}>
      {onBack && (
        <div style={{ marginBottom: 10 }}>
          <GhostBtn icon={ICONS.arrowL} label="Zurück" onClick={onBack} />
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
        <div>
          <div className="eyebrow" style={{ fontSize: 12, marginBottom: 4 }}>Spielbericht</div>
          <h2 style={{ margin: 0, fontSize: 27, lineHeight: 1.08, fontFamily: 'var(--font-heading)', letterSpacing: '-0.01em' }}>{headline}</h2>
        </div>
        {onReplay && <GhostBtn icon={ICONS.replay} label="Animationen wiederholen" onClick={onReplay} />}
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 13, color: 'var(--muted)' }}>
        {m.weekday ? `${m.weekday}, ` : ''}
        {m.dateLabel} · {disciplineDE(m.discipline)}
        {m.court ? ` · Feld ${m.court}` : ''}
      </p>
      <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <ResultChip result={m.result} line={m.resultLine} />
        {m.score && <span style={{ fontSize: 13, color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>{m.score}</span>}
      </div>
    </header>
  )
}
