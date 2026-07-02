import { describe, expect, it } from 'vitest'
import { buildRallySavePlan, type RallyReviewRow } from './rallyPlan'
import { buildRallyCorrections } from './rallyCorrections'

const NOW = '2026-07-02T12:00:00.000Z'

function row(id: string, idx: number, over: Partial<RallyReviewRow> = {}): RallyReviewRow {
  return {
    id,
    idx,
    tStart: idx * 10,
    tEnd: idx * 10 + 5,
    shotCount: 3,
    winnerSide: null,
    endShotType: null,
    confidence: 0.8,
    editedAt: null,
    ...over,
  }
}

describe('buildRallyCorrections', () => {
  it('returns no rows when nothing changed', () => {
    const rows = [row('a', 0), row('b', 1)]
    const plan = buildRallySavePlan(rows, new Set(), new Set(), NOW)
    expect(buildRallyCorrections(rows, plan)).toEqual([])
  })

  it('records a drop with the pre-edit snapshot', () => {
    const rows = [row('a', 0), row('b', 1, { tStart: 8, tEnd: 12, shotCount: 5, confidence: 0.4 })]
    const plan = buildRallySavePlan(rows, new Set(['b']), new Set(), NOW)
    expect(buildRallyCorrections(rows, plan)).toEqual([
      {
        op: 'drop',
        rally_id: 'b',
        rally_idx: 1,
        merged_into: null,
        orig_t_start_s: 8,
        orig_t_end_s: 12,
        orig_shot_count: 5,
        orig_winner_side: null,
        orig_end_shot_type: null,
        orig_confidence: 0.4,
        patch: null,
      },
    ])
  })

  it('records an adjacent merge as one absorbed row plus one keep row', () => {
    const a = row('a', 0, { tStart: 1, tEnd: 4, shotCount: 3, confidence: 0.9 })
    const b = row('b', 1, { tStart: 8, tEnd: 12, shotCount: 5, confidence: 0.6, winnerSide: 'B', endShotType: 'smash' })
    const plan = buildRallySavePlan([a, b], new Set(), new Set(['b']), NOW)
    const cors = buildRallyCorrections([a, b], plan)

    const absorbed = cors.find((c) => c.op === 'merge_absorbed')
    expect(absorbed).toMatchObject({
      rally_id: 'b',
      rally_idx: 1,
      merged_into: 'a',
      orig_t_start_s: 8,
      orig_t_end_s: 12,
      patch: null,
    })

    const keep = cors.find((c) => c.op === 'merge_keep')
    // the keep row snapshots its PRE-edit state; patch is exactly what the plan wrote
    expect(keep).toMatchObject({
      rally_id: 'a',
      rally_idx: 0,
      merged_into: null,
      orig_t_end_s: 4,
      orig_shot_count: 3,
      orig_confidence: 0.9,
      patch: plan.rallyUpdates[0].patch,
    })
    expect(cors).toHaveLength(2)
  })

  it('records a 3-row merge chain as two absorbed rows and one keep', () => {
    const rows = [row('a', 0), row('b', 1), row('c', 2)]
    const plan = buildRallySavePlan(rows, new Set(), new Set(['b', 'c']), NOW)
    const cors = buildRallyCorrections(rows, plan)
    expect(cors.filter((c) => c.op === 'merge_absorbed').map((c) => c.rally_id)).toEqual(['b', 'c'])
    expect(cors.filter((c) => c.op === 'merge_absorbed').every((c) => c.merged_into === 'a')).toBe(true)
    expect(cors.filter((c) => c.op === 'merge_keep').map((c) => c.rally_id)).toEqual(['a'])
  })

  it('records an independent drop and merge together', () => {
    const rows = [row('a', 0), row('b', 1), row('c', 2)]
    const plan = buildRallySavePlan(rows, new Set(['c']), new Set(['b']), NOW)
    const cors = buildRallyCorrections(rows, plan)
    expect(cors.map((c) => [c.op, c.rally_id]).sort()).toEqual([
      ['drop', 'c'],
      ['merge_absorbed', 'b'],
      ['merge_keep', 'a'],
    ])
  })
})
