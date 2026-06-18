import type { ReportPlayer } from '../../types'
import { projectUV } from '../../lib/homography'
import { trackAt, type TrackKey } from '../../lib/overlay'
import type { OverlayToggles } from '../../hooks/useVideoPlayer'

interface Props {
  time: number // seconds (real video.currentTime, or the synthetic demo clock)
  players: ReportPlayer[]
  ov: OverlayToggles
  H: number[] // court(u,v) → frame fraction (real calibration or demo fallback)
  tracks: Map<string, TrackKey[]> // by subject 'player:<id>' | 'shuttle'; t in seconds
}

const VB_W = 192
const VB_H = 108
const ORANGE = '#FA8832'
const BLUE = '#1758A6'

const clamp = (x: number) => Math.max(0, Math.min(1, x))

/**
 * Shared SVG analysis-overlay engine over the video surface (mobile card + desktop Safari frame).
 * Geometry is computed in court coordinates and projected via the real homography (lib/homography.ts);
 * player positions come from real track_keyframes (P1) or the demo fallback (seeded matches). The
 * nearness term is `v` itself (v=0 far baseline … v=1 near baseline) — calibration-independent.
 */
export function VideoOverlay({ time, players, ov, H, tracks }: Props) {
  // Court (u,v) → viewBox px, with a nearness term `t` for perspective sizing.
  const frame = (u: number, v: number) => {
    const p = projectUV(H, u, v)
    return { x: p.x * VB_W, y: p.y * VB_H, t: clamp(v) }
  }

  const viewerSide = players[0]?.side
  const keysFor = (p: ReportPlayer) => tracks.get(`player:${p.playerId}`) ?? []
  const colorFor = (p: ReportPlayer) => (p.side === viewerSide ? ORANGE : BLUE)
  const labelFor = (p: ReportPlayer) => (p.isViewer ? 'Du' : p.first)

  const playerBox = (pos: { u: number; v: number }, name: string, color: string) => {
    const pt = frame(pos.u, pos.v)
    const w = 4.5 + pt.t * 4.5
    const h = 8 + pt.t * 7
    const chipW = Math.max(w, name.length * 2.0 + 2.4)
    return (
      <g>
        <rect x={pt.x - w / 2} y={pt.y - h} width={w} height={h} rx="1" fill={color + '14'} stroke={color} strokeWidth="0.45" />
        <rect x={pt.x - w / 2} y={pt.y - h - 4.4} width={chipW} height="3.6" rx="1" fill={color} />
        <text x={pt.x - w / 2 + 1.2} y={pt.y - h - 1.7} fontSize="2.7" fontWeight="700" fill="#FFFFFF" style={{ fontFamily: 'var(--font-body)' }}>
          {name}
        </text>
        <ellipse cx={pt.x} cy={pt.y} rx={w * 0.62} ry="1.1" fill={color} opacity="0.35" />
      </g>
    )
  }

  const trail = (keys: TrackKey[], color: string) => {
    const pts: string[] = []
    for (let dt = 5; dt >= 0; dt -= 0.25) {
      const pos = trackAt(keys, time - dt) // trackAt clamps before the first / after the last key
      const fr = frame(pos.u, pos.v)
      pts.push(`${fr.x.toFixed(1)},${fr.y.toFixed(1)}`)
    }
    return (
      <g fill="none" strokeLinecap="round" strokeLinejoin="round">
        <polyline points={pts.slice(0, 11).join(' ')} stroke={color} strokeWidth="0.55" opacity="0.25" />
        <polyline points={pts.slice(10).join(' ')} stroke={color} strokeWidth="0.8" opacity="0.65" />
      </g>
    )
  }

  const zones = () => {
    const z = players[0]?.stats.zones
    const bands = [
      { v0: 0.5, v1: 0.65, pct: z?.front ?? 0 },
      { v0: 0.65, v1: 0.82, pct: z?.mid ?? 0 },
      { v0: 0.82, v1: 1.0, pct: z?.rear ?? 0 },
    ]
    return (
      <g>
        {bands.map((b, i) => {
          const tl = frame(0, b.v0)
          const tr = frame(1, b.v0)
          const bl = frame(0, b.v1)
          const br = frame(1, b.v1)
          return (
            <g key={i}>
              <polygon
                points={`${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`}
                fill={ORANGE}
                opacity={0.1 + (b.pct / 100) * 0.5}
                stroke="#FFFFFF"
                strokeWidth="0.25"
                strokeOpacity="0.5"
              />
              <text
                x={(tr.x + br.x) / 2 + 2.5}
                y={(tr.y + br.y) / 2 + 1}
                fontSize="3"
                fontWeight="700"
                fill="#FFFFFF"
                stroke="rgba(0,0,0,.35)"
                strokeWidth="0.12"
                style={{ fontFamily: 'var(--font-head)' }}
              >
                {b.pct}%
              </text>
            </g>
          )
        })}
      </g>
    )
  }

  // Players that actually have a track this frame (real CV, or the demo fallback). Draw far→near.
  const tracked = players
    .map((p) => ({ p, keys: keysFor(p) }))
    .filter((x) => x.keys.length > 0)
    .map((x) => ({ ...x, pos: trackAt(x.keys, time) }))
    .sort((a, b) => a.pos.v - b.pos.v)

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {ov.zonen && zones()}
      {ov.wege && tracked.map(({ p, keys }) => <g key={`t${p.playerId}`}>{trail(keys, colorFor(p))}</g>)}
      {/* P1 has no shuttle tracking — the shuttle layer renders only once a 'shuttle' track exists. */}
      {ov.tracking && tracked.map(({ p, pos }) => <g key={`b${p.playerId}`}>{playerBox(pos, labelFor(p), colorFor(p))}</g>)}
    </svg>
  )
}
