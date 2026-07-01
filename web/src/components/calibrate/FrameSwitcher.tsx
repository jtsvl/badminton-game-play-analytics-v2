import { useEffect } from 'react'
import { fmtTime } from '../../lib/format'

interface Props {
  count: number
  index: number
  seconds: number
  onIndex: (i: number) => void
}

const btn = (disabled: boolean): React.CSSProperties => ({
  padding: '8px 12px',
  borderRadius: 10,
  border: '1px solid var(--hairline)',
  background: 'var(--card, #fff)',
  color: 'var(--ink)',
  fontWeight: 700,
  fontSize: 14,
  fontFamily: 'var(--font-body)',
  cursor: disabled ? 'not-allowed' : 'pointer',
  opacity: disabled ? 0.5 : 1,
  whiteSpace: 'nowrap',
})

/**
 * Step between the calibration snapshots so a corner blocked by a player in one frame can be placed
 * on another (the court is static → switching the background never moves the handles). Mirrors the
 * shuttle labeler's controls: ◀ Zurück · slider · Weiter ▶ · „Frame i/N · m:ss" · ←/→ keys.
 */
export function FrameSwitcher({ count, index, seconds, onIndex }: Props) {
  const clampTo = (i: number) => onIndex(Math.max(0, Math.min(count - 1, i)))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Let the range input handle its own arrow keys when it's focused (avoid double-stepping).
      if (e.target instanceof HTMLInputElement && e.target.type === 'range') return
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        clampTo(index + 1)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        clampTo(index - 1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, count])

  if (count <= 1) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <button type="button" style={btn(index <= 0)} disabled={index <= 0} onClick={() => clampTo(index - 1)}>
          ◀ Zurück
        </button>
        <input
          type="range"
          min={0}
          max={count - 1}
          value={index}
          onChange={(e) => clampTo(Number(e.target.value))}
          style={{ flex: 1, minWidth: 160, accentColor: 'var(--crusta, #FA8832)' }}
          aria-label="Standbild wählen"
        />
        <button type="button" style={btn(index >= count - 1)} disabled={index >= count - 1} onClick={() => clampTo(index + 1)}>
          Weiter ▶
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'var(--font-body)' }}>
        Bild {index + 1}/{count} · {fmtTime(seconds)} · Wähle einen Moment, in dem alle vier Ecken frei sind.
      </div>
    </div>
  )
}
