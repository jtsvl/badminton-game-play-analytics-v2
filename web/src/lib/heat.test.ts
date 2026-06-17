import { describe, it, expect } from 'vitest'
import { heatGrid, type HeatSpot } from './heat'

describe('heatGrid', () => {
  it('produces cols×rows cells', () => {
    const g = heatGrid([{ x: 0.5, y: 0.5, w: 1, s: 0.2 }], 12, 13)
    expect(g).toHaveLength(12 * 13)
  })

  it('normalises so the peak cell is 1.0 and all cells ≤ 1', () => {
    const g = heatGrid([{ x: 0.5, y: 0.5, w: 1, s: 0.2 }], 12, 13)
    const max = Math.max(...g.map((c) => c.v))
    expect(max).toBeCloseTo(1, 6)
    expect(g.every((c) => c.v <= 1 + 1e-9 && c.v >= 0)).toBe(true)
  })

  it('puts the peak cell nearest the spot centre', () => {
    const spot: HeatSpot = { x: 0.5, y: 0.5, w: 1, s: 0.15 }
    const g = heatGrid([spot], 12, 13)
    const peak = g.reduce((a, b) => (b.v > a.v ? b : a))
    const cx = (peak.c + 0.5) / 12
    const cy = (peak.r + 0.5) / 13
    expect(Math.abs(cx - spot.x)).toBeLessThan(1 / 12)
    expect(Math.abs(cy - spot.y)).toBeLessThan(1 / 13)
  })

  it('drives far cells below the 0.05 render threshold', () => {
    const g = heatGrid([{ x: 0.5, y: 0.5, w: 1, s: 0.1 }], 12, 13)
    const corner = g.find((c) => c.c === 0 && c.r === 0)!
    expect(corner.v).toBeLessThan(0.05)
  })

  it('handles empty spots without dividing by zero', () => {
    const g = heatGrid([], 12, 13)
    expect(g.every((c) => c.v === 0)).toBe(true)
  })
})
