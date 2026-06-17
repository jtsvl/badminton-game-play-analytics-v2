import type { ReportPlayer } from '../../types'

interface Props {
  players: ReportPlayer[]
  tab: number
  setTab: (i: number) => void
}

/** Segmented player selector (theme A). Viewer → „Du", others → first name. */
export function PlayerTabs({ players, tab, setTab }: Props) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 4,
        padding: 4,
        borderRadius: 12,
        background: 'rgba(10,10,11,.05)',
        border: '1px solid var(--hairline)',
      }}
    >
      {players.map((p, i) => (
        <button
          key={p.name + i}
          type="button"
          onClick={() => setTab(i)}
          style={{
            flex: 1,
            border: 'none',
            borderRadius: 9,
            padding: '8px 4px',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 600,
            fontFamily: 'var(--font-body)',
            background: tab === i ? 'var(--card)' : 'transparent',
            color: tab === i ? 'var(--ink)' : 'var(--muted)',
            boxShadow: tab === i ? 'var(--shadow-sm)' : 'none',
            transition: 'background .2s ease, color .2s ease',
          }}
        >
          {p.isViewer ? 'Du' : p.first}
        </button>
      ))}
    </div>
  )
}
