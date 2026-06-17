import { useId, useMemo } from 'react'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'
import { heatGrid, type HeatSpot } from '../../lib/heat'

interface Props {
  spots: HeatSpot[]
  /** Only the brand `raster` style is shipped (specs/50-web-app.md). */
  style?: 'raster'
  width?: number
  anim?: Anim
}

// Half court, top-down, net at top. 1 m = 40 px → 244 × 268 (court 6.1 × 6.7 m).
const W = 244
const H = 268
const COLS = 12
const ROWS = 13

/** Player-coverage heatmap: 12×13 raster of gaussian kernels; hottest (>85 %) cells turn red. */
export function CourtHeatmap({ spots, width = 222, anim = 'dezent' }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.25)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  // useId returns ":r0:"-style ids; strip non-alphanumerics so url(#…) is valid and unique per instance.
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '')
  const grid = useMemo(() => heatGrid(spots, COLS, ROWS), [spots])
  const cw = W / COLS
  const ch = H / ROWS

  return (
    <div ref={ref} style={{ lineHeight: 0 }}>
      <svg width={width} height={(width * H) / W} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <defs>
          <clipPath id={`clip${uid}`}>
            <rect x="0" y="0" width={W} height={H} />
          </clipPath>
        </defs>
        <rect x="0" y="0" width={W} height={H} fill="#FFFFFF" />
        <g clipPath={`url(#clip${uid})`}>
          {grid.map((cell) =>
            cell.v > 0.05 ? (
              <rect
                key={`${cell.r}-${cell.c}`}
                x={cell.c * cw + 1.6}
                y={cell.r * ch + 1.6}
                width={cw - 3.2}
                height={ch - 3.2}
                rx="3"
                fill={cell.v > 0.85 ? '#C0392B' : '#FA8832'}
                style={{
                  opacity: on ? Math.min(0.95, 0.07 + cell.v * 0.88) : 0,
                  transition: `opacity .45s ease ${(cell.c + cell.r) * 22}ms`,
                }}
              />
            ) : null,
          )}
        </g>
        <g stroke="#DDD5CC" strokeWidth="2" fill="none">
          <rect x="1" y="1" width={W - 2} height={H - 2} />
          <line x1="18.4" y1="1" x2="18.4" y2={H - 1} />
          <line x1={W - 18.4} y1="1" x2={W - 18.4} y2={H - 1} />
          <line x1="1" y1="79" x2={W - 1} y2="79" />
          <line x1="1" y1={H - 30} x2={W - 1} y2={H - 30} />
          <line x1={W / 2} y1="79" x2={W / 2} y2={H - 1} />
          <line x1="0" y1="3" x2={W} y2="3" stroke="#B9AFA4" strokeWidth="4" />
        </g>
      </svg>
    </div>
  )
}
