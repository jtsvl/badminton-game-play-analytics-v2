import type { ReportPlayer } from '../../types'
import { projectUV } from '../../lib/homography'
import { trackAt, shuttleAt } from '../../lib/overlay'
import { H_DEMO, DEMO_TRACK_NEAR, DEMO_TRACK_FAR, DEMO_LOOP, FAR_Y, NEAR_Y } from '../../lib/demoOverlay'
import type { OverlayToggles } from '../../hooks/useVideoPlayer'

interface Props {
  time: number
  players: ReportPlayer[]
  ov: OverlayToggles
}

const VB_W = 192
const VB_H = 108
const ORANGE = '#FA8832'
const BLUE = '#1758A6'

const clamp = (x: number) => Math.max(0, Math.min(1, x))

// Court (u,v) → viewBox px, with a nearness term `t` (0 far … 1 near) for perspective sizing.
function frame(u: number, v: number) {
  const p = projectUV(H_DEMO, u, v)
  return { x: p.x * VB_W, y: p.y * VB_H, t: clamp((p.y - FAR_Y) / (NEAR_Y - FAR_Y)) }
}

/**
 * Shared SVG analysis-overlay engine over the video surface (mobile card + desktop Safari frame).
 * Geometry is computed in court coordinates and projected via the real homography (lib/homography.ts);
 * tracks are demo data for now (lib/demoOverlay.ts) — swap to track_keyframes + real calibration in P1.
 */
export function VideoOverlay({ time, players, ov }: Props) {
  const p = (time % DEMO_LOOP) / DEMO_LOOP
  const near = players[0]
  const far = players[1]
  const nearPos = trackAt(DEMO_TRACK_NEAR, p)
  const farPos = trackAt(DEMO_TRACK_FAR, p)

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

  const trail = (keys: typeof DEMO_TRACK_NEAR, color: string) => {
    const pts: string[] = []
    for (let dt = 5; dt >= 0; dt -= 0.25) {
      let tp = ((time - dt) % DEMO_LOOP) / DEMO_LOOP
      if (tp < 0) tp += 1
      const pos = trackAt(keys, tp)
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
    const z = near?.stats.zones
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

  const shuttle = () => {
    const s = shuttleAt(p, DEMO_TRACK_NEAR, DEMO_TRACK_FAR)
    const cur = frame(s.u, s.v)
    const from = frame(s.from.u, s.from.v)
    return (
      <g>
        <line x1={from.x} y1={from.y - 4} x2={cur.x} y2={cur.y - 2} stroke="#FFFFFF" strokeWidth="0.4" strokeDasharray="1.3 1" opacity="0.7" />
        <circle cx={cur.x} cy={cur.y - 2 - Math.sin(Math.PI * s.segP) * 6} r="1" fill="#FFFFFF" stroke="rgba(0,0,0,.3)" strokeWidth="0.2" />
      </g>
    )
  }

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {ov.zonen && zones()}
      {ov.wege && trail(DEMO_TRACK_NEAR, ORANGE)}
      {ov.wege && trail(DEMO_TRACK_FAR, BLUE)}
      {ov.shuttle && shuttle()}
      {ov.tracking && far && playerBox(farPos, far.first, BLUE)}
      {ov.tracking && near && playerBox(nearPos, near.isViewer ? 'Du' : near.first, ORANGE)}
    </svg>
  )
}
