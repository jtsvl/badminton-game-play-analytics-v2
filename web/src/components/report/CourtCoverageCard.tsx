import type { CSSProperties } from 'react'
import type { HeatSpot } from '../../lib/heat'
import type { Anim } from '../../lib/motion'
import { SectionCard } from './SectionCard'
import { CourtHeatmap } from './CourtHeatmap'
import { HeatLegend } from './HeatLegend'

interface Props {
  spots: HeatSpot[]
  sub: string
  width?: number
  dense?: boolean
  /** Grow the heatmap to fill remaining card height (desktop right column). */
  fill?: boolean
  style?: CSSProperties
  anim?: Anim
}

/** "Feldabdeckung" card — the court heatmap + legend, used by both layouts. */
export function CourtCoverageCard({ spots, sub, width = 222, dense, fill, style, anim }: Props) {
  return (
    <SectionCard title="Feldabdeckung" sub={sub} dense={dense} style={style}>
      <div
        style={{
          flex: fill ? 1 : undefined,
          minHeight: fill ? 0 : undefined,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: fill ? 0 : '4px 0 0',
        }}
      >
        <CourtHeatmap spots={spots} width={width} anim={anim} />
      </div>
      <HeatLegend />
    </SectionCard>
  )
}
