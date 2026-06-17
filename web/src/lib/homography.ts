// Planar homography — the production court→frame projection engine for video overlays.
// Replaces the prototype's hand-tuned perspective quad (video.jsx `VID_QUAD`/`vidProject`).
// A badminton court is a plane, so a single 3×3 homography is the correct projective model.
// In P1 the matrix comes from `court_calibrations.homography`; for the demo it is derived from
// four court-corner↔frame correspondences (see lib/demoOverlay.ts).

export type Pt = [number, number]

/** Solve a dense linear system A·x = b (Gaussian elimination, partial pivoting). */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length
  // augmented matrix
  const m = A.map((row, i) => [...row, b[i]])
  for (let col = 0; col < n; col++) {
    // pivot: largest magnitude in this column
    let piv = col
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r
    if (Math.abs(m[piv][col]) < 1e-12) throw new Error('homography: singular system')
    ;[m[col], m[piv]] = [m[piv], m[col]]
    // eliminate below
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = m[r][col] / m[col][col]
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c]
    }
  }
  return m.map((row, i) => row[n] / row[i])
}

/**
 * Solve the 3×3 homography H mapping `src[i]` → `dst[i]` for four point pairs.
 * Returns the 9 entries [h0..h8] row-major, normalised with h8 = 1.
 */
export function solveHomography(src: Pt[], dst: Pt[]): number[] {
  if (src.length !== 4 || dst.length !== 4) throw new Error('homography: need exactly 4 point pairs')
  const A: number[][] = []
  const b: number[] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i]
    const [X, Y] = dst[i]
    A.push([x, y, 1, 0, 0, 0, -x * X, -y * X])
    b.push(X)
    A.push([0, 0, 0, x, y, 1, -x * Y, -y * Y])
    b.push(Y)
  }
  const h = solveLinear(A, b)
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]
}

/** Project court coordinates (u,v) through homography H to frame coordinates {x,y}. */
export function projectUV(H: number[], u: number, v: number): { x: number; y: number } {
  const denom = H[6] * u + H[7] * v + H[8]
  return {
    x: (H[0] * u + H[1] * v + H[2]) / denom,
    y: (H[3] * u + H[4] * v + H[5]) / denom,
  }
}
