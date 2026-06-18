import { describe, it, expect } from 'vitest'
import { projectUV } from './homography'
import { COURT_CORNERS, homographyFromCorners, isConvexQuad } from './calibration'

describe('homographyFromCorners', () => {
  // A pixel rectangle inset 10% in a 1000x500 frame -> court corners map to those fractions.
  const size = { w: 1000, h: 500 }
  const cornersPx: [number, number][] = [
    [100, 50], // FL
    [900, 50], // FR
    [900, 450], // NR
    [100, 450], // NL
  ]

  it('maps each court corner to its frame fraction', () => {
    const H = homographyFromCorners(cornersPx, size)
    const expected = [
      [0.1, 0.1],
      [0.9, 0.1],
      [0.9, 0.9],
      [0.1, 0.9],
    ]
    COURT_CORNERS.forEach((c, i) => {
      const p = projectUV(H, c[0], c[1])
      expect(p.x).toBeCloseTo(expected[i][0], 6)
      expect(p.y).toBeCloseTo(expected[i][1], 6)
    })
  })

  it('maps court centre to frame centre for this affine inset', () => {
    const H = homographyFromCorners(cornersPx, size)
    const p = projectUV(H, 0.5, 0.5)
    expect(p.x).toBeCloseTo(0.5, 6)
    expect(p.y).toBeCloseTo(0.5, 6)
  })
})

describe('isConvexQuad', () => {
  it('accepts a proper convex quad', () => {
    expect(isConvexQuad([[0, 0], [1, 0], [1, 1], [0, 1]])).toBe(true)
    // a perspective trapezoid (near edge wider) is still convex
    expect(isConvexQuad([[0.4, 0.4], [0.6, 0.4], [0.72, 0.95], [0.28, 0.95]])).toBe(true)
  })

  it('rejects a self-intersecting (bowtie) quad', () => {
    expect(isConvexQuad([[0, 0], [1, 1], [1, 0], [0, 1]])).toBe(false)
  })

  it('rejects a degenerate (collinear) quad', () => {
    expect(isConvexQuad([[0, 0], [1, 0], [2, 0], [3, 0]])).toBe(false)
  })
})
