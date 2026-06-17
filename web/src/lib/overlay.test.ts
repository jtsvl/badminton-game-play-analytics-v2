import { describe, it, expect } from 'vitest'
import { trackAt, shuttleAt, type TrackKey } from './overlay'

const LINE: TrackKey[] = [
  { t: 0, u: 0, v: 0 },
  { t: 1, u: 10, v: 0 },
]

describe('trackAt', () => {
  it('clamps to the first key at or before the start', () => {
    expect(trackAt(LINE, 0)).toEqual({ u: 0, v: 0 })
    expect(trackAt(LINE, -0.5)).toEqual({ u: 0, v: 0 })
  })

  it('clamps to the last key at or beyond the end', () => {
    expect(trackAt(LINE, 1)).toEqual({ u: 10, v: 0 })
    expect(trackAt(LINE, 2)).toEqual({ u: 10, v: 0 })
  })

  it('applies smoothstep between keys', () => {
    expect(trackAt(LINE, 0.5).u).toBeCloseTo(5, 6) // smoothstep(0.5) = 0.5
    expect(trackAt(LINE, 0.25).u).toBeCloseTo(1.5625, 6) // 0.25^2*(3-0.5)
  })

  it('selects the right segment with three keys', () => {
    const keys: TrackKey[] = [
      { t: 0, u: 0, v: 0 },
      { t: 0.5, u: 4, v: 0 },
      { t: 1, u: 4, v: 8 },
    ]
    expect(trackAt(keys, 0.5)).toEqual({ u: 4, v: 0 }) // boundary returns that key
    expect(trackAt(keys, 0.75).v).toBeCloseTo(4, 6) // smoothstep(0.5)=0.5 of 0→8
  })
})

describe('shuttleAt', () => {
  const near: TrackKey[] = [
    { t: 0, u: 0.5, v: 0.8 },
    { t: 1, u: 0.5, v: 0.8 },
  ]
  const far: TrackKey[] = [
    { t: 0, u: 0.5, v: 0.2 },
    { t: 1, u: 0.5, v: 0.2 },
  ]

  it('starts a near→far segment at the near player', () => {
    const s = shuttleAt(0, near, far)
    expect(s.from.v).toBeCloseTo(0.8, 6) // segment 0 starts from near side
    expect(s.segP).toBeCloseTo(0, 6)
  })

  it('reaches the far player midway through the first segment', () => {
    const s = shuttleAt(0.05, near, far) // 10 segments → seg 0, segP 0.5
    expect(s.segP).toBeCloseTo(0.5, 6)
    expect(s.v).toBeCloseTo(0.5, 6) // halfway from near(0.8) to far(0.2)
  })
})
