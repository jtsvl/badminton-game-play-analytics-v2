import type { CSSProperties } from 'react'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'

interface Props {
  points: number[]
  sub?: string
  dense?: boolean
  /** Cap the chart width inside a narrow tile (desktop). */
  chartMaxWidth?: number
  style?: CSSProperties
  anim?: Anim
}

const W = 330
const H = 120
const BASE = 96

/** "Aktivität" — distance per 5-minute bucket as a bar chart; the peak bucket is labelled. */
export function ActivityCard({ points, sub = 'Distanz je 5 Minuten', dense, chartMaxWidth, style, anim = 'dezent' }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.3)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  const n = points.length || 1
  const max = Math.max(...points, 1)
  const maxI = points.indexOf(Math.max(...points))
  const bw = W / n

  return (
    <SectionCard title="Aktivität" sub={sub} dense={dense} style={style}>
      <div ref={ref} style={{ maxWidth: chartMaxWidth, margin: chartMaxWidth ? '0 auto' : undefined }}>
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
          {points.map((v, i) => {
            const h = (v / max) * 82
            const hot = i === maxI
            return (
              <g key={i}>
                <rect
                  x={i * bw + 3}
                  y={BASE - h}
                  width={bw - 6}
                  height={h}
                  rx={4}
                  fill={hot ? '#FA8832' : 'rgba(250,136,50,.42)'}
                  style={{
                    transform: on ? 'scaleY(1)' : 'scaleY(0)',
                    transformOrigin: 'bottom',
                    transformBox: 'fill-box',
                    transition: `transform .55s cubic-bezier(.22,.8,.3,1) ${i * 55}ms`,
                  }}
                />
                {hot && (
                  <text
                    x={i * bw + bw / 2}
                    y={BASE - h - 7}
                    textAnchor="middle"
                    fontSize="10.5"
                    fontWeight="700"
                    fill="#0A0A0B"
                    fontFamily="var(--font-head)"
                    style={{ opacity: on ? 1 : 0, transition: `opacity .4s ease ${300 + i * 55}ms` }}
                  >
                    {v}&#8201;m
                  </text>
                )}
                {i % 2 === 0 && (
                  <text x={i * bw + bw / 2} y={H - 8} textAnchor="middle" fontSize="9.5" fill="hsl(215.4 16.3% 46.9%)">
                    {i * 5}&#8242;
                  </text>
                )}
              </g>
            )
          })}
          <line x1="0" y1={BASE + 1} x2={W} y2={BASE + 1} stroke="rgba(10,10,11,.12)" strokeWidth="1" />
        </svg>
      </div>
    </SectionCard>
  )
}
