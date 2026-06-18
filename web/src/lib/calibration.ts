// Calibration glue: derive the court→frame-fraction homography from the 4 confirmed corners, reusing
// the production engine (lib/homography.ts) with ZERO new matrix math. The 4 corners (+ image_size)
// are the single source of truth: the worker derives image-px→(u,v) for its geometry; the web derives
// the inverse direction here for overlay rendering. See specs/features/frontend-port.md (P1 swap-in).

import { solveHomography, type Pt } from './homography'
import { H_DEMO, DEMO_TRACK_NEAR, DEMO_TRACK_FAR, DEMO_LOOP } from './demoOverlay'
import type { TrackKey } from './overlay'
import type { MatchReport } from '../types'

// Court corners in (u,v): far-left, far-right, near-right, near-left — matches the worker's
// confirmed_points order and the canonical demoOverlay COURT_CORNERS.
export const COURT_CORNERS: Pt[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
]

export interface ImageSize {
  w: number
  h: number
}

/** Pixel corners (FL,FR,NR,NL) + image size → court(u,v)→frame-fraction homography (9 row-major). */
export function homographyFromCorners(cornersPx: Pt[], size: ImageSize): number[] {
  const frac: Pt[] = cornersPx.map(([x, y]) => [x / size.w, y / size.h])
  return solveHomography(COURT_CORNERS, frac)
}

/** True when the four corners form a simple convex quad (guards the homography solve + Save button). */
export function isConvexQuad(pts: Pt[]): boolean {
  if (pts.length !== 4) return false
  let sign = 0
  for (let i = 0; i < 4; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % 4]
    const c = pts[(i + 2) % 4]
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    if (Math.abs(cross) < 1e-9) return false // collinear / degenerate
    const s = Math.sign(cross)
    if (sign === 0) sign = s
    else if (s !== sign) return false // a reflex turn => non-convex / self-intersecting
  }
  return true
}

export interface OverlayInputs {
  H: number[] // court(u,v) → frame-fraction
  tracks: Map<string, TrackKey[]> // by subject 'player:<id>' | 'shuttle'; t in SECONDS
  isDemo: boolean
}

/**
 * Resolve the overlay's geometry + tracks for a report. Calibrated → real homography from the
 * confirmed corners plus whatever track_keyframes exist (possibly none yet). Uncalibrated (seeded
 * demo matches) → the demo homography + demo tracks scaled to seconds, keyed by the two players'
 * ids, so the existing P0 demo keeps animating. `trackAt` is unit-agnostic, so callers always pass
 * real video time in seconds.
 */
export function overlayInputs(report: MatchReport): OverlayInputs {
  const calib = report.calib
  if (calib?.confirmedPoints && calib.imageSize) {
    const H = homographyFromCorners(calib.confirmedPoints, calib.imageSize)
    return { H, tracks: new Map(report.tracks.map((t) => [t.subject, t.data])), isDemo: false }
  }
  const toSeconds = (keys: TrackKey[]): TrackKey[] =>
    keys.map((k) => ({ t: k.t * DEMO_LOOP, u: k.u, v: k.v }))
  const tracks = new Map<string, TrackKey[]>()
  if (report.players[0]) tracks.set(`player:${report.players[0].playerId}`, toSeconds(DEMO_TRACK_NEAR))
  if (report.players[1]) tracks.set(`player:${report.players[1].playerId}`, toSeconds(DEMO_TRACK_FAR))
  return { H: H_DEMO, tracks, isDemo: true }
}
