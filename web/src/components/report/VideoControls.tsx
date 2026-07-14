import type { CSSProperties } from 'react'
import { fmtTime } from '../../lib/format'
import type { VideoPlayer } from '../../hooks/useVideoPlayer'

interface Props {
  player: VideoPlayer
  style?: CSSProperties
  /** Hit-editor opt-in: show a slow-mo speed selector + playhead frame-step for precise annotation. */
  showRate?: boolean
  /** Video fps for frame-accurate stepping (defaults to 30 if unknown). */
  fps?: number
}

const RATES = [0.25, 0.5, 1, 2]
const rateLabel = (r: number) => `${String(r).replace('.', ',')}×` // de-DE decimal comma

/** Play/pause bubble button + scrubber + time readout. With `showRate`, adds a speed selector and
 * ◀/▶ 1-frame buttons that step the PAUSED video one frame (precise hit annotation). */
export function VideoControls({ player, style, showRate, fps }: Props) {
  const mainRow = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <button
        type="button"
        onClick={player.togglePlay}
        aria-label={player.playing ? 'Pause' : 'Abspielen'}
        style={{
          width: 38, height: 38, borderRadius: 19, border: '1px solid #C66C27', cursor: 'pointer',
          background: '#FA8832', color: '#1A0500', display: 'flex', alignItems: 'center',
          justifyContent: 'center', boxShadow: '0 3px 0 #C66C27', flex: 'none', padding: 0,
        }}
      >
        {player.playing ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <rect x="5" y="4" width="5" height="16" rx="1.2" />
            <rect x="14" y="4" width="5" height="16" rx="1.2" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 4.8v14.4c0 .9 1 1.5 1.8 1L20 13c.8-.5.8-1.6 0-2.1L8.8 3.9C8 3.4 7 4 7 4.8z" />
          </svg>
        )}
      </button>
      <input
        type="range"
        min={0}
        max={player.dur || 0}
        step={0.1}
        value={player.time}
        onChange={(e) => player.seek(parseFloat(e.target.value))}
        className="bca-range"
        style={{ flex: 1, accentColor: '#FA8832' }}
      />
      <span style={{ fontSize: 11.5, color: 'var(--muted)', fontVariantNumeric: 'tabular-nums', fontFamily: 'var(--font-mono)', flex: 'none' }}>
        {fmtTime(player.time)} / {fmtTime(player.dur)}
      </span>
    </div>
  )

  if (!showRate) return <div style={style}>{mainRow}</div>

  const stepFrame = (dir: number) => {
    if (player.playing) player.togglePlay() // frame-stepping only makes sense paused
    const f = 1 / (fps || 30)
    player.seek(Math.max(0, Math.min(player.dur || 0, player.time + dir * f)))
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, ...style }}>
      {mainRow}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12, color: 'var(--muted)', marginRight: 2 }}>Tempo</span>
        {RATES.map((r) => {
          const active = player.rate === r
          return (
            <button
              key={r}
              type="button"
              aria-label={`Tempo ${rateLabel(r)}`}
              aria-pressed={active}
              onClick={() => player.setRate(r)}
              style={{ ...rateBtn, ...(active ? rateBtnActive : null) }}
            >
              {rateLabel(r)}
            </button>
          )
        })}
        <span style={{ display: 'inline-flex', gap: 6, marginLeft: 'auto' }}>
          <button type="button" aria-label="Video 1 Frame zurück" onClick={() => stepFrame(-1)} style={rateBtn}>◀ 1F</button>
          <button type="button" aria-label="Video 1 Frame vor" onClick={() => stepFrame(1)} style={rateBtn}>1F ▶</button>
        </span>
      </div>
    </div>
  )
}

const rateBtn: CSSProperties = {
  border: '1px solid #C66C27', background: '#FEF4EA', color: '#1A0500', borderRadius: 8,
  padding: '3px 9px', fontSize: 12.5, cursor: 'pointer', fontFamily: 'var(--font-body)',
  fontVariantNumeric: 'tabular-nums',
}
const rateBtnActive: CSSProperties = { background: '#FA8832', boxShadow: '0 2px 0 #C66C27' }
