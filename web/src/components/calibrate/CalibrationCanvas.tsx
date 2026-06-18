import { useRef, useState } from 'react'
import type { Pt } from '../../lib/homography'
import { projectUV } from '../../lib/homography'
import { COURT_TEMPLATE } from '../../lib/courtTemplate'

interface Props {
  imgSrc: string | null
  size: { w: number; h: number }
  corners: Pt[] // image px, order FL, FR, NR, NL
  H: number[] // court(u,v) → frame fraction, derived from `corners`
  valid: boolean
  onDrag: (i: number, px: Pt) => void
}

const ORANGE = '#FA8832'
const BLUE = '#1758A6'
const LABELS = ['FL', 'FR', 'NR', 'NL']
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

/** The still + an SVG overlay: live-projected court template + four draggable corner handles. */
export function CalibrationCanvas({ imgSrc, size, corners, H, valid, onDrag }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [active, setActive] = useState<number | null>(null)
  // Handle radius in image-px units (the SVG viewBox is image px); scale with frame so it feels
  // consistent on a 4K still.
  const r = Math.max(size.w, size.h) * 0.011
  const stroke = r * 0.28

  const toPx = (clientX: number, clientY: number): Pt => {
    const rect = boxRef.current!.getBoundingClientRect()
    return [
      clamp(((clientX - rect.left) / rect.width) * size.w, 0, size.w),
      clamp(((clientY - rect.top) / rect.height) * size.h, 0, size.h),
    ]
  }

  // Project a court (u,v) endpoint to image px through the live homography.
  const toImg = (u: number, v: number) => {
    const p = projectUV(H, u, v)
    return { x: p.x * size.w, y: p.y * size.h }
  }

  const lineColor = valid ? ORANGE : '#C0392B'

  return (
    <div
      ref={boxRef}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: `${size.w} / ${size.h}`,
        background: '#0A0A0B',
        borderRadius: 10,
        overflow: 'hidden',
        touchAction: 'none',
        userSelect: 'none',
      }}
    >
      {imgSrc && (
        <img
          src={imgSrc}
          alt="Kalibrierungs-Standbild"
          draggable={false}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block', pointerEvents: 'none' }}
        />
      )}
      <svg
        viewBox={`0 0 ${size.w} ${size.h}`}
        preserveAspectRatio="none"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none' }}
      >
        {/* projected court template (drag until it sits on the painted lines) */}
        <g fill="none" stroke={lineColor} strokeOpacity="0.95" strokeLinecap="round">
          {COURT_TEMPLATE.map(([a, b], i) => {
            const p1 = toImg(a[0], a[1])
            const p2 = toImg(b[0], b[1])
            const outer = i < 4 // the outer doubles boundary, drawn heavier
            return (
              <g key={i}>
                <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#000" strokeOpacity="0.35" strokeWidth={(outer ? r * 0.5 : r * 0.34) + stroke} />
                <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} strokeWidth={outer ? r * 0.5 : r * 0.34} />
              </g>
            )
          })}
        </g>

        {/* corner handles */}
        {corners.map((c, i) => {
          const near = i >= 2
          const color = near ? ORANGE : BLUE
          const isActive = active === i
          return (
            <g
              key={i}
              style={{ cursor: 'grab', touchAction: 'none' }}
              onPointerDown={(e) => {
                try {
                  ;(e.target as Element).setPointerCapture(e.pointerId)
                } catch {
                  /* capture is best-effort */
                }
                setActive(i)
              }}
              onPointerMove={(e) => {
                if (active === i) onDrag(i, toPx(e.clientX, e.clientY))
              }}
              onPointerUp={(e) => {
                try {
                  ;(e.target as Element).releasePointerCapture(e.pointerId)
                } catch {
                  /* noop */
                }
                setActive(null)
              }}
            >
              {/* generous transparent hit target for touch */}
              <circle cx={c[0]} cy={c[1]} r={r * 2.4} fill="transparent" />
              {isActive && (
                <>
                  <line x1={c[0]} y1={0} x2={c[0]} y2={size.h} stroke={color} strokeWidth={stroke * 0.5} strokeOpacity="0.5" />
                  <line x1={0} y1={c[1]} x2={size.w} y2={c[1]} stroke={color} strokeWidth={stroke * 0.5} strokeOpacity="0.5" />
                </>
              )}
              <circle cx={c[0]} cy={c[1]} r={isActive ? r * 1.25 : r} fill={color + '33'} stroke={color} strokeWidth={stroke} />
              <circle cx={c[0]} cy={c[1]} r={r * 0.18} fill={color} />
              <text x={c[0] + r * 1.4} y={c[1] - r * 1.1} fontSize={r * 1.5} fontWeight="700" fill={color} stroke="#000" strokeOpacity="0.4" strokeWidth={r * 0.04} style={{ fontFamily: 'var(--font-head)' }}>
                {LABELS[i]}
              </text>
            </g>
          )
        })}

        {/* Magnifier: while dragging, a zoomed loupe (with crosshair + projected grid) in the corner
            diagonally opposite the handle, so the corner can be placed exactly on a painted line. */}
        {active !== null &&
          (() => {
            const Z = 3.4 // zoom factor
            const R = Math.max(size.w, size.h) * 0.15 // loupe radius (image px)
            const pad = R + Math.max(size.w, size.h) * 0.02
            const [hx, hy] = corners[active]
            const lcx = hx < size.w / 2 ? size.w - pad : pad // opposite corner → never under the finger
            const lcy = hy < size.h / 2 ? size.h - pad : pad
            const tx = lcx - hx * Z
            const ty = lcy - hy * Z
            const cross = R * 0.5
            return (
              <g pointerEvents="none">
                <defs>
                  <clipPath id="calib-loupe-clip">
                    <circle cx={lcx} cy={lcy} r={R} />
                  </clipPath>
                </defs>
                <g clipPath="url(#calib-loupe-clip)">
                  <rect x={lcx - R} y={lcy - R} width={2 * R} height={2 * R} fill="#0A0A0B" />
                  <g transform={`translate(${tx} ${ty}) scale(${Z})`}>
                    {imgSrc && (
                      <image href={imgSrc} x={0} y={0} width={size.w} height={size.h} preserveAspectRatio="none" />
                    )}
                    <g fill="none" stroke={lineColor} strokeOpacity="0.95" strokeLinecap="round">
                      {COURT_TEMPLATE.map(([a, b], i) => {
                        const p1 = toImg(a[0], a[1])
                        const p2 = toImg(b[0], b[1])
                        return (
                          <line key={i} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} strokeWidth={(i < 4 ? r * 0.5 : r * 0.34) / Z} />
                        )
                      })}
                    </g>
                  </g>
                  {/* crosshair at the exact handle point */}
                  <line x1={lcx - cross} y1={lcy} x2={lcx + cross} y2={lcy} stroke="#FFFFFF" strokeWidth={stroke * 0.5} strokeOpacity="0.9" />
                  <line x1={lcx} y1={lcy - cross} x2={lcx} y2={lcy + cross} stroke="#FFFFFF" strokeWidth={stroke * 0.5} strokeOpacity="0.9" />
                  <circle cx={lcx} cy={lcy} r={r * 0.5} fill="none" stroke="#FFFFFF" strokeWidth={stroke * 0.4} strokeOpacity="0.65" />
                </g>
                <circle cx={lcx} cy={lcy} r={R} fill="none" stroke="#000" strokeOpacity="0.3" strokeWidth={stroke * 1.8} />
                <circle cx={lcx} cy={lcy} r={R} fill="none" stroke="#FFFFFF" strokeWidth={stroke} />
              </g>
            )
          })()}
      </svg>
    </div>
  )
}
