// Rally-correction rows (FR-35 eval corpus) — pure mapper, no Supabase calls. Combines the
// pre-edit row snapshot with the already-computed RallySavePlan into DB-column-shaped rows for
// `rally_corrections` (0011 migration). The web never writes that table: the rows ride the
// existing rally_review job as params.corrections and the worker's rally_sync stage inserts
// them server-side.

import type { RallyPatch, RallyReviewRow, RallySavePlan } from './rallyPlan'

export interface RallyCorrectionRow {
  op: 'drop' | 'merge_absorbed' | 'merge_keep'
  rally_id: string
  rally_idx: number
  merged_into: string | null
  orig_t_start_s: number | null
  orig_t_end_s: number | null
  orig_shot_count: number | null
  orig_winner_side: 'A' | 'B' | null
  orig_end_shot_type: string | null
  orig_confidence: number | null
  patch: RallyPatch | null
}

export function buildRallyCorrections(rows: RallyReviewRow[], plan: RallySavePlan): RallyCorrectionRow[] {
  const byId = new Map(rows.map((r) => [r.id, r]))
  const keepIdByAbsorbed = new Map(plan.eventReparents.map((rp) => [rp.from, rp.to]))
  const out: RallyCorrectionRow[] = []
  const origOf = (r: RallyReviewRow) => ({
    orig_t_start_s: r.tStart, orig_t_end_s: r.tEnd, orig_shot_count: r.shotCount,
    orig_winner_side: r.winnerSide, orig_end_shot_type: r.endShotType, orig_confidence: r.confidence,
  })
  for (const d of plan.rallyDeletes) {
    const row = byId.get(d.id)
    if (!row) continue
    out.push({
      op: d.deleteEvents ? 'drop' : 'merge_absorbed',
      rally_id: d.id, rally_idx: row.idx,
      merged_into: d.deleteEvents ? null : (keepIdByAbsorbed.get(d.id) ?? null),
      patch: null, ...origOf(row),
    })
  }
  for (const u of plan.rallyUpdates) {
    const row = byId.get(u.id)
    if (!row) continue
    out.push({ op: 'merge_keep', rally_id: u.id, rally_idx: row.idx, merged_into: null, patch: u.patch, ...origOf(row) })
  }
  return out
}
