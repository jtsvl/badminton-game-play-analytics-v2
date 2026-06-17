import { describe, it, expect } from 'vitest'
import { solveHomography, projectUV, type Pt } from './homography'

const SQUARE: Pt[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
]

describe('homography', () => {
  it('identity: unit square → unit square maps (u,v) to itself', () => {
    const H = solveHomography(SQUARE, SQUARE)
    for (const [u, v] of [
      [0.5, 0.5],
      [0.2, 0.8],
      [0, 0],
      [1, 1],
    ]) {
      const p = projectUV(H, u, v)
      expect(p.x).toBeCloseTo(u, 6)
      expect(p.y).toBeCloseTo(v, 6)
    }
  })

  it('affine (scale + translate) is exact at an interior point', () => {
    // x → 2u + 10, y → v + 20  (a homography with h6=h7=0)
    const dst: Pt[] = [
      [10, 20],
      [12, 20],
      [12, 21],
      [10, 21],
    ]
    const H = solveHomography(SQUARE, dst)
    const p = projectUV(H, 0.5, 0.5)
    expect(p.x).toBeCloseTo(11, 6)
    expect(p.y).toBeCloseTo(20.5, 6)
  })

  it('round-trips every corner of a perspective trapezoid', () => {
    const trap: Pt[] = [
      [0.4, 0.415],
      [0.615, 0.415],
      [0.72, 0.955],
      [0.34, 0.955],
    ]
    const H = solveHomography(SQUARE, trap)
    SQUARE.forEach((corner, i) => {
      const p = projectUV(H, corner[0], corner[1])
      expect(p.x).toBeCloseTo(trap[i][0], 6)
      expect(p.y).toBeCloseTo(trap[i][1], 6)
    })
  })
})
