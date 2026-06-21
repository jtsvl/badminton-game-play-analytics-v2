import { projectUV } from '../../lib/homography'
import { type TrackKey } from '../../lib/overlay'
import type { ReviewRow } from './types'

interface Props {
  H: number[] | null // court(u,v) → frame fraction (from the confirmed corners); null → no overlay
  frameUrl: string | null
  size: { w: number; h: number }
  rows: ReviewRow[]
  pathByLabel: Map<number, TrackKey[]> // detected track id → its keyframes (intrinsic to the track)
  focused: number | null // index into rows whose path to emphasise
}

const ORANGE = '#FA8832' // side A ("our" side)
const BLUE = '#1758A6' // side B (opponent)
const VB_W = 192
const VB_H = 108

const colorFor = (side: ReviewRow['side']) => (side === 'A' ? ORANGE : BLUE)

/**
 * Read-only still-frame overlay that draws each detected track's full path so the trainer can tell who
 * is who before confirming the mapping. Reuses the report overlay's engine (lib/homography.projectUV);
 * static here (whole path, not animated). A path is intrinsic to a track_label, so rows render the path
 * of their currently-selected track. The focused row is emphasised, the rest dimmed.
 */
export function PlayerReviewOverlay({ H, frameUrl, size, rows, pathByLabel, focused }: Props) {
  const aspect = size.h > 0 ? size.w / size.h : VB_W / VB_H
  const project = (u: number, v: number) => {
    const p = projectUV(H as number[], u, v)
    return { x: p.x * VB_W, y: p.y * VB_H }
  }
  const polyline = (keys: TrackKey[]) =>
    keys.map((k) => { const p = project(k.u, k.v); return `${p.x.toFixed(1)},${p.y.toFixed(1)}` }).join(' ')

  // Rows that actually have a track + path this run (real CV; empty when nothing computed yet).
  const drawable = H
    ? rows
        .map((r, i) => ({ r, i, keys: r.trackLabel !== null ? pathByLabel.get(r.trackLabel) ?? [] : [] }))
        .filter((x) => x.keys.length > 0)
    : []

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: String(aspect),
        background: frameUrl ? `#000 url(${frameUrl}) center/cover no-repeat` : 'var(--card, #fff)',
        border: '1px solid var(--hairline, hsl(214.3 31.8% 91.4%))',
        borderRadius: 'var(--radius-card, 14px)',
        overflow: 'hidden',
      }}
    >
      {drawable.length > 0 ? (
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
        >
          {drawable.map(({ r, i, keys }) => {
            const on = focused === null || focused === i
            const color = colorFor(r.side)
            const start = project(keys[0].u, keys[0].v)
            const end = project(keys[keys.length - 1].u, keys[keys.length - 1].v)
            return (
              <g key={r.id} opacity={on ? 1 : 0.25}>
                <polyline
                  points={polyline(keys)}
                  fill="none"
                  stroke={color}
                  strokeWidth={on ? 1.1 : 0.7}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity={on ? 0.85 : 0.6}
                />
                <circle cx={start.x} cy={start.y} r={on ? 1.4 : 1} fill="#fff" stroke={color} strokeWidth="0.5" />
                <circle cx={end.x} cy={end.y} r={on ? 2.2 : 1.6} fill={color} />
                <text
                  x={end.x + 2.5}
                  y={end.y + 1}
                  fontSize="3.4"
                  fontWeight="700"
                  fill="#fff"
                  stroke="rgba(0,0,0,.35)"
                  strokeWidth="0.12"
                  style={{ fontFamily: 'var(--font-head, Space Grotesk)' }}
                >
                  {`#${r.trackLabel}`}
                </text>
              </g>
            )
          })}
        </svg>
      ) : (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--muted, hsl(215.4 16.3% 46.9%))',
            fontSize: 13,
            textAlign: 'center',
            padding: 16,
          }}
        >
          {H ? 'Noch keine Laufwege berechnet.' : 'Keine Kalibrierung – Wege können nicht angezeigt werden.'}
        </div>
      )}
    </div>
  )
}
