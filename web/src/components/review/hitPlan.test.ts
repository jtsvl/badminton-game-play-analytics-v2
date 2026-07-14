import { describe, expect, it } from 'vitest'
import { buildHitSavePlan, hitValidationError, type HitReviewRow } from './hitPlan'

function hit(id: string, tS: number, over: Partial<HitReviewRow> = {}): HitReviewRow {
  return { id, tS, rallyId: 'r1', rallyIdx: 1, confidence: 0.3, payload: null, ...over }
}

describe('buildHitSavePlan', () => {
  it('no edits ⇒ no changes', () => {
    const orig = [hit('a', 10), hit('b', 12)]
    const plan = buildHitSavePlan(orig, [hit('a', 10), hit('b', 12)])
    expect(plan.hasChanges).toBe(false)
    expect(plan.inserts).toEqual([])
    expect(plan.nudges).toEqual([])
    expect(plan.deletes).toEqual([])
  })

  it('an isNew row ⇒ an insert carrying its temp id + rally', () => {
    const orig = [hit('a', 10)]
    const edited = [hit('a', 10), hit('tmp-1', 11.5, { isNew: true, rallyId: 'r2' })]
    const plan = buildHitSavePlan(orig, edited)
    expect(plan.inserts).toEqual([{ tempId: 'tmp-1', t_s: 11.5, rally_id: 'r2' }])
    expect(plan.hasChanges).toBe(true)
  })

  it('an original missing from edited ⇒ a delete with its orig time', () => {
    const orig = [hit('a', 10), hit('b', 12)]
    const plan = buildHitSavePlan(orig, [hit('a', 10)])
    expect(plan.deletes).toEqual([{ id: 'b', orig_t_s: 12 }])
  })

  it('a changed t_s beyond the epsilon ⇒ a nudge; within it ⇒ ignored', () => {
    const orig = [hit('a', 10), hit('b', 12)]
    const edited = [hit('a', 10.2), hit('b', 12.004)] // a moved 200ms, b moved 4ms (< 8ms eps)
    const plan = buildHitSavePlan(orig, edited)
    expect(plan.nudges).toEqual([{ id: 'a', t_s: 10.2, orig_t_s: 10 }])
    expect(plan.hasChanges).toBe(true)
  })

  it('add + delete + nudge together', () => {
    const orig = [hit('a', 10), hit('b', 12), hit('c', 14)]
    const edited = [hit('a', 10.3), hit('c', 14), hit('tmp-1', 16, { isNew: true })] // b deleted, a nudged, tmp added
    const plan = buildHitSavePlan(orig, edited)
    expect(plan.inserts.map((i) => i.tempId)).toEqual(['tmp-1'])
    expect(plan.deletes.map((d) => d.id)).toEqual(['b'])
    expect(plan.nudges.map((n) => n.id)).toEqual(['a'])
  })
})

describe('hitValidationError', () => {
  it('accepts a normal list', () => {
    expect(hitValidationError([hit('a', 0), hit('b', 421.5)])).toBeNull()
  })
  it('rejects a negative or non-finite time', () => {
    expect(hitValidationError([hit('a', -1)])).not.toBeNull()
    expect(hitValidationError([hit('a', NaN)])).not.toBeNull()
  })
})
