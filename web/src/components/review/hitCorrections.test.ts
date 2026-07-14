import { describe, expect, it } from 'vitest'
import { buildHitSavePlan, type HitReviewRow } from './hitPlan'
import { buildHitCorrections } from './hitCorrections'

function hit(id: string, tS: number, over: Partial<HitReviewRow> = {}): HitReviewRow {
  return { id, tS, rallyId: 'r1', rallyIdx: 1, confidence: 0.3, payload: null, ...over }
}

describe('buildHitCorrections', () => {
  it('maps add/delete/nudge to the correct op rows with before/after times', () => {
    const orig = [hit('a', 10, { confidence: 0.5 }), hit('b', 12, { payload: { shot_type: 'smash' } })]
    const edited = [
      hit('a', 10.4), // nudge
      // b deleted
      hit('tmp-1', 16, { isNew: true, rallyId: 'r2', rallyIdx: 2 }), // add
    ]
    const plan = buildHitSavePlan(orig, edited)
    const rows = buildHitCorrections(orig, edited, plan, new Map([['tmp-1', 'evt-99']]))

    const add = rows.find((r) => r.op === 'add')!
    expect(add).toMatchObject({ event_id: 'evt-99', rally_id: 'r2', rally_idx: 2, orig_t_s: null, new_t_s: 16 })

    const nudge = rows.find((r) => r.op === 'nudge')!
    expect(nudge).toMatchObject({ event_id: 'a', orig_t_s: 10, new_t_s: 10.4, orig_confidence: 0.5 })

    const del = rows.find((r) => r.op === 'delete')!
    expect(del).toMatchObject({ event_id: 'b', orig_t_s: 12, new_t_s: null, orig_payload: { shot_type: 'smash' } })
  })

  it('skips an add whose insert id did not resolve (insert failed)', () => {
    const orig: HitReviewRow[] = []
    const edited = [hit('tmp-1', 5, { isNew: true })]
    const plan = buildHitSavePlan(orig, edited)
    const rows = buildHitCorrections(orig, edited, plan, new Map()) // empty ⇒ unresolved
    expect(rows).toEqual([])
  })
})
