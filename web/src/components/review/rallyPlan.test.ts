import { describe, expect, it } from 'vitest'
import { buildRallySavePlan, mergeValidationError, type RallyReviewRow } from './rallyPlan'

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

describe('mergeValidationError', () => {
  it('accepts no flags', () => {
    expect(mergeValidationError([row('a', 0), row('b', 1)], new Set(), new Set())).toBeNull()
  })

  it('rejects merge on the first rally (no predecessor)', () => {
    expect(mergeValidationError([row('a', 0), row('b', 1)], new Set(), new Set(['a']))).toMatch(/erste/)
  })

  it('rejects merging into a dropped predecessor', () => {
    expect(mergeValidationError([row('a', 0), row('b', 1)], new Set(['a']), new Set(['b']))).toMatch(/entfernt/)
  })

  it('rejects a row that is both dropped and merged', () => {
    expect(mergeValidationError([row('a', 0), row('b', 1)], new Set(['b']), new Set(['b']))).toMatch(/entfernt/)
  })

  it('accepts a chain of merges next to an unrelated drop', () => {
    const rows = [row('a', 0), row('b', 1), row('c', 2), row('d', 3)]
    expect(mergeValidationError(rows, new Set(['d']), new Set(['b', 'c']))).toBeNull()
  })
})

describe('buildRallySavePlan', () => {
  it('reports no changes when nothing is flagged', () => {
    const plan = buildRallySavePlan([row('a', 0)], new Set(), new Set(), NOW)
    expect(plan.hasChanges).toBe(false)
    expect(plan.rallyUpdates).toEqual([])
    expect(plan.eventReparents).toEqual([])
    expect(plan.rallyDeletes).toEqual([])
  })

  it('drops a rally with its events', () => {
    const plan = buildRallySavePlan([row('a', 0), row('b', 1)], new Set(['b']), new Set(), NOW)
    expect(plan.hasChanges).toBe(true)
    expect(plan.rallyDeletes).toEqual([{ id: 'b', deleteEvents: true }])
    expect(plan.rallyUpdates).toEqual([])
  })

  it('merges an adjacent pair: span, sum, min confidence, end fields from the second', () => {
    const a = row('a', 0, { tStart: 1, tEnd: 4, shotCount: 3, confidence: 0.9 })
    const b = row('b', 1, {
      tStart: 8,
      tEnd: 12,
      shotCount: 5,
      confidence: 0.6,
      winnerSide: 'B',
      endShotType: 'smash',
    })
    const plan = buildRallySavePlan([a, b], new Set(), new Set(['b']), NOW)
    expect(plan.rallyUpdates).toEqual([
      {
        id: 'a',
        patch: {
          t_end_s: 12,
          shot_count: 8,
          winner_side: 'B',
          end_shot_type: 'smash',
          confidence: 0.6,
          edited_at: NOW,
        },
      },
    ])
    // events move BEFORE the absorbed row is deleted (FK is ON DELETE SET NULL)
    expect(plan.eventReparents).toEqual([{ from: 'b', to: 'a' }])
    expect(plan.rallyDeletes).toEqual([{ id: 'b', deleteEvents: false }])
  })

  it('folds a consecutive chain into one merge group', () => {
    const rows = [
      row('a', 0, { tEnd: 5, shotCount: 2, confidence: 0.9 }),
      row('b', 1, { shotCount: null, confidence: 0.7 }),
      row('c', 2, { tEnd: 25, shotCount: 4, confidence: 0.8, endShotType: 'clear' }),
    ]
    const plan = buildRallySavePlan(rows, new Set(), new Set(['b', 'c']), NOW)
    expect(plan.rallyUpdates).toHaveLength(1)
    expect(plan.rallyUpdates[0].id).toBe('a')
    expect(plan.rallyUpdates[0].patch.t_end_s).toBe(25)
    expect(plan.rallyUpdates[0].patch.shot_count).toBe(6) // null counts as 0
    expect(plan.rallyUpdates[0].patch.confidence).toBe(0.7)
    expect(plan.rallyUpdates[0].patch.end_shot_type).toBe('clear')
    expect(plan.eventReparents).toEqual([
      { from: 'b', to: 'a' },
      { from: 'c', to: 'a' },
    ])
    expect(plan.rallyDeletes).toEqual([
      { id: 'b', deleteEvents: false },
      { id: 'c', deleteEvents: false },
    ])
  })

  it('keeps null shot_count when no member has one', () => {
    const rows = [row('a', 0, { shotCount: null }), row('b', 1, { shotCount: null })]
    const plan = buildRallySavePlan(rows, new Set(), new Set(['b']), NOW)
    expect(plan.rallyUpdates[0].patch.shot_count).toBeNull()
  })

  it('combines an independent drop with a merge', () => {
    const rows = [row('a', 0), row('b', 1), row('c', 2)]
    const plan = buildRallySavePlan(rows, new Set(['c']), new Set(['b']), NOW)
    expect(plan.rallyUpdates.map((u) => u.id)).toEqual(['a'])
    expect(plan.rallyDeletes).toEqual([
      { id: 'b', deleteEvents: false },
      { id: 'c', deleteEvents: true },
    ])
  })
})
