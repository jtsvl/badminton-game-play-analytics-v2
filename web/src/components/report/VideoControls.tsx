import type { CSSProperties } from 'react'
import { fmtTime } from '../../lib/format'
import type { VideoPlayer } from '../../hooks/useVideoPlayer'

interface Props {
  player: VideoPlayer
  style?: CSSProperties
}

/** Play/pause bubble button + scrubber + time readout. */
export function VideoControls({ player, style }: Props) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, ...style }}>
      <button
        type="button"
        onClick={player.togglePlay}
        aria-label={player.playing ? 'Pause' : 'Abspielen'}
        style={{
          width: 38,
          height: 38,
          borderRadius: 19,
          border: '1px solid #C66C27',
          cursor: 'pointer',
          background: '#FA8832',
          color: '#1A0500',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 3px 0 #C66C27',
          flex: 'none',
          padding: 0,
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
}
