import { describe, it, expect } from 'vitest'
import { poseAt, type PoseKey } from './pose'

// poseAt only picks a frame — one joint is enough to identify which key was returned.
const key = (t: number, x: number): PoseKey => ({ t, kp: [[x, 0.5, 0.9]] })

const KEYS: PoseKey[] = [key(0, 0.1), key(0.2, 0.2), key(0.4, 0.3)]

describe('poseAt', () => {
  it('returns null for empty keys', () => {
    expect(poseAt([], 1)).toBeNull()
  })

  it('picks the nearest keyframe, never interpolating', () => {
    expect(poseAt(KEYS, 0.05)?.kp[0][0]).toBeCloseTo(0.1, 6) // closer to t=0
    expect(poseAt(KEYS, 0.15)?.kp[0][0]).toBeCloseTo(0.2, 6) // closer to t=0.2
    expect(poseAt(KEYS, 0.2)?.kp[0][0]).toBeCloseTo(0.2, 6) // exact hit
  })

  it('resolves an exact midpoint to the later key', () => {
    expect(poseAt(KEYS, 0.1)?.t).toBeCloseTo(0.2, 6) // 0.1 is equidistant from t=0 and t=0.2
  })

  it('clamps to the first/last key within maxGapS', () => {
    expect(poseAt(KEYS, -0.2)?.t).toBeCloseTo(0, 6) // 0.2 s before the start
    expect(poseAt(KEYS, 0.65)?.t).toBeCloseTo(0.4, 6) // 0.25 s past the end
  })

  it('returns null beyond maxGapS (out-of-rally / track gap → hide, do not freeze)', () => {
    expect(poseAt(KEYS, -0.31)).toBeNull()
    expect(poseAt(KEYS, 0.75)).toBeNull()
    expect(poseAt([key(0, 0.1), key(1, 0.2)], 0.5)).toBeNull() // gap in the middle
  })

  it('honors a custom maxGapS', () => {
    expect(poseAt(KEYS, 0.65, 0.1)).toBeNull()
    expect(poseAt(KEYS, 0.65, 0.25)?.t).toBeCloseTo(0.4, 6) // exactly maxGapS away still shows
  })
})
