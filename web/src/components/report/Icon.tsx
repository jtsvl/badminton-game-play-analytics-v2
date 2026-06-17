import type { CSSProperties, ReactNode } from 'react'

interface Props {
  d: ReactNode
  size?: number
  color?: string
  fill?: string
  style?: CSSProperties
}

/** Tiny inline stroke icon on a 24-grid (ported from the handoff — no icon dependency). */
export function Icon({ d, size = 18, color = 'currentColor', fill = 'none', style }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={fill}
      stroke={color}
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ display: 'block', ...style }}
    >
      {d}
    </svg>
  )
}

export const ICONS: Record<string, ReactNode> = {
  route: (
    <g>
      <circle cx="5" cy="19" r="2" />
      <circle cx="19" cy="5" r="2" />
      <path d="M7 19h7a4 4 0 0 0 0-8H10a4 4 0 0 1 0-8h7" />
    </g>
  ),
  gauge: (
    <g>
      <path d="M12 15l3.5-3.5" />
      <path d="M20.3 14a8.5 8.5 0 1 0-16.6 0" />
    </g>
  ),
  bolt: (
    <g>
      <path d="M13 2 4.5 14H11l-1 8 8.5-12H12l1-8z" />
    </g>
  ),
  clock: (
    <g>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </g>
  ),
  chevR: <path d="M9 6l6 6-6 6" />,
  arrowL: <path d="M19 12H5m6-7-7 7 7 7" />,
  replay: (
    <g>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </g>
  ),
  chevD: <path d="M6 9l6 6 6-6" />,
}
