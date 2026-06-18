import { describe, it, expect } from 'vitest'
import { COURT_TEMPLATE, SINGLES_X, SHORT_SERVICE_DV, NET_V } from './courtTemplate'

describe('courtTemplate', () => {
  it('places the net across the middle (v=0.5)', () => {
    expect(NET_V).toBeCloseTo(0.5, 6)
    const net = COURT_TEMPLATE.find(
      ([a, b]) => a[1] === NET_V && b[1] === NET_V && a[0] === 0 && b[0] === 1,
    )
    expect(net).toBeDefined()
  })

  it('uses BWF-correct singles inset and short-service offset (normalised over 13.40 m)', () => {
    expect(SINGLES_X).toBeCloseTo(0.46 / 6.1, 4) // ~0.0754
    expect(SHORT_SERVICE_DV).toBeCloseTo(1.98 / 13.4, 4) // ~0.1478
  })

  it('keeps every segment endpoint inside the unit court', () => {
    for (const [a, b] of COURT_TEMPLATE) {
      for (const [u, v] of [a, b]) {
        expect(u).toBeGreaterThanOrEqual(0)
        expect(u).toBeLessThanOrEqual(1)
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThanOrEqual(1)
      }
    }
  })

  it('includes the four outer doubles boundary lines (a closed quad)', () => {
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    for (let i = 0; i < 4; i++) {
      const a = corners[i]
      const b = corners[(i + 1) % 4]
      const found = COURT_TEMPLATE.some(
        ([s, e]) =>
          (s[0] === a[0] && s[1] === a[1] && e[0] === b[0] && e[1] === b[1]) ||
          (s[0] === b[0] && s[1] === b[1] && e[0] === a[0] && e[1] === a[1]),
      )
      expect(found).toBe(true)
    }
  })
})
