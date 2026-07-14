// Hit-correction rows (the G0 eval + training corpus) — pure mapper, no Supabase calls. Combines
// the pre-edit snapshot + the computed HitSavePlan into DB-column-shaped rows for `hit_corrections`
// (0013 migration). The web never writes that table: the rows ride a `hit_review` job as
// params.corrections and the worker's hit_review stage inserts them server-side. Mirrors
// ./rallyCorrections.ts. `insertedIds` maps each insert's temp id to the real events.id the INSERT
// returned, so an 'add' correction points at a durable row.

import type { HitReviewRow, HitSavePlan } from './hitPlan'

export interface HitCorrectionRow {
  op: 'add' | 'delete' | 'nudge'
  event_id: string
  rally_id: string | null
  rally_idx: number | null
  orig_t_s: number | null
  new_t_s: number | null
  orig_confidence: number | null
  orig_payload: unknown | null
}

export function buildHitCorrections(
  original: HitReviewRow[],
  edited: HitReviewRow[],
  plan: HitSavePlan,
  insertedIds: Map<string, string>,
): HitCorrectionRow[] {
  const byOrig = new Map(original.map((o) => [o.id, o]))
  const byEdited = new Map(edited.map((e) => [e.id, e]))
  const out: HitCorrectionRow[] = []

  for (const ins of plan.inserts) {
    const eventId = insertedIds.get(ins.tempId)
    if (!eventId) continue // insert failed / id not resolved — skip rather than log a bad row
    const e = byEdited.get(ins.tempId)
    out.push({
      op: 'add',
      event_id: eventId,
      rally_id: e?.rallyId ?? ins.rally_id,
      rally_idx: e?.rallyIdx ?? null,
      orig_t_s: null,
      new_t_s: ins.t_s,
      orig_confidence: null,
      orig_payload: null,
    })
  }

  for (const n of plan.nudges) {
    const o = byOrig.get(n.id)
    out.push({
      op: 'nudge',
      event_id: n.id,
      rally_id: o?.rallyId ?? null,
      rally_idx: o?.rallyIdx ?? null,
      orig_t_s: n.orig_t_s,
      new_t_s: n.t_s,
      orig_confidence: o?.confidence ?? null,
      orig_payload: o?.payload ?? null,
    })
  }

  for (const d of plan.deletes) {
    const o = byOrig.get(d.id)
    out.push({
      op: 'delete',
      event_id: d.id,
      rally_id: o?.rallyId ?? null,
      rally_idx: o?.rallyIdx ?? null,
      orig_t_s: d.orig_t_s,
      new_t_s: null,
      orig_confidence: o?.confidence ?? null,
      orig_payload: o?.payload ?? null,
    })
  }

  return out
}
