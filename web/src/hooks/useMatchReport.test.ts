// useMatchReport wiring test — locks the result derivation at the hook level: the canonical
// matches.result_side_a drives meta.result/resultLine (no "Sieg 0:0" on empty games), flipped for a
// side-B viewer. The result logic itself is TDD'd in lib/score.test.ts; this proves the hook feeds it
// the DB column. Minimal thenable-builder stub for supabase-js (query builders are awaitable).
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useMatchReport } from './useMatchReport'

const h = vi.hoisted(() => ({ tables: {} as Record<string, { data: unknown }> }))

function stub(result: { data: unknown }) {
  const o: Record<string, unknown> = {}
  for (const m of ['select', 'eq', 'order', 'limit']) o[m] = () => o
  const resolved = Promise.resolve({ error: null, ...result })
  o.single = () => resolved
  o.maybeSingle = () => resolved
  o.then = resolved.then.bind(resolved)
  return o
}

vi.mock('../lib/supabase', () => ({
  isConfigured: true,
  supabase: { from: (table: string) => stub(h.tables[table] ?? { data: [] }) },
}))
vi.mock('../lib/storage', () => ({ resolveVideoSrc: async () => '' }))

function seedTables() {
  h.tables = {
    matches: {
      data: {
        id: 'm1', slug: 'test', played_on: '2026-06-10', weekday: 'Mi', discipline: 'singles',
        court: 1, score: null, games: [], duration_min: null, rest_min: null,
        result_side_a: 'win', status: 'done',
      },
    },
    match_players: {
      data: [
        { side: 'A', slot: 1, player_id: 'p1', players: { name: 'Ada', first_name: 'A' } },
        { side: 'B', slot: 1, player_id: 'p2', players: { name: 'Ben', first_name: 'B' } },
      ],
    },
    match_rally_stats: { data: null },
  }
}

describe('useMatchReport result wiring', () => {
  beforeEach(seedTables)

  it('uses canonical result_side_a with empty games (no "Sieg 0:0")', async () => {
    const { result } = renderHook(() => useMatchReport('test'))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.error).toBeNull()
    expect(result.current.report?.meta.result).toBe('win')
    expect(result.current.report?.meta.resultLine).toBe('Sieg')
  })

  it('flips the canonical result for a side-B viewer', async () => {
    const { result } = renderHook(() => useMatchReport('test', 'p2'))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.report?.meta.result).toBe('loss')
    expect(result.current.report?.meta.resultLine).toBe('Niederlage')
    expect(result.current.report?.players[0]?.playerId).toBe('p2') // viewer sorted first
  })
})
