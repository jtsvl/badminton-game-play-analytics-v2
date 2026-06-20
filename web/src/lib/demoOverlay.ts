// DEMO overlay input — the intentional FALLBACK for uncalibrated (seeded) matches, feeding the real
// projection engine (lib/homography.ts). The live path landed in P1 (commit 22fc7e7): lib/calibration.ts
// `overlayInputs` derives the court→frame homography from `court_calibrations.confirmed_points` and uses
// the real `track_keyframes` whenever a calibration exists, only dropping to this demo otherwise. Note we
// do NOT read `court_calibrations.homography` — that stored matrix is the inverse direction (image-px→u,v,
// for the worker's gating); the web re-derives the court→frame-fraction direction from the 4 corners.
// This is the only place the prototype's hand-tuned geometry survives, still as a canonical homography.

import { solveHomography, type Pt } from './homography'
import type { TrackKey } from './overlay'

// Court corners in (u,v): far-left, far-right, near-right, near-left.
const COURT_CORNERS: Pt[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
]

// Their positions as fractions of the demo video frame, taken from the prototype baseline quad
// (video.jsx VID_QUAD: far baseline y=0.415, near baseline y=0.955).
export const FAR_Y = 0.415
export const NEAR_Y = 0.955
const FRAME_CORNERS: Pt[] = [
  [0.4, FAR_Y],
  [0.615, FAR_Y],
  [0.72, NEAR_Y],
  [0.34, NEAR_Y],
]

/** Demo homography: court (u,v) → frame fraction (0..1). Computed once at module load. */
export const H_DEMO = solveHomography(COURT_CORNERS, FRAME_CORNERS)

/** Demo movement loop length in seconds; the synthetic clock (no real footage) spans this. */
export const DEMO_LOOP = 16

const key = ([t, u, v]: [number, number, number]): TrackKey => ({ t, u, v })

// Near (viewer) and far (opponent) player tracks across one loop.
export const DEMO_TRACK_NEAR: TrackKey[] = (
  [
    [0, 0.5, 0.8],
    [0.12, 0.32, 0.66],
    [0.22, 0.6, 0.62],
    [0.34, 0.52, 0.9],
    [0.46, 0.7, 0.7],
    [0.58, 0.44, 0.62],
    [0.7, 0.3, 0.82],
    [0.82, 0.62, 0.84],
    [1, 0.5, 0.8],
  ] as [number, number, number][]
).map(key)

export const DEMO_TRACK_FAR: TrackKey[] = (
  [
    [0, 0.5, 0.22],
    [0.1, 0.66, 0.36],
    [0.2, 0.4, 0.14],
    [0.32, 0.56, 0.3],
    [0.44, 0.34, 0.4],
    [0.56, 0.6, 0.12],
    [0.68, 0.46, 0.34],
    [0.8, 0.64, 0.2],
    [0.9, 0.4, 0.3],
    [1, 0.5, 0.22],
  ] as [number, number, number][]
).map(key)
