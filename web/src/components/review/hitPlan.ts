// Hit-review save plan — pure logic for the /match/:slug/hits hit-editor (stroke-classification-v2
// §5 Stage 1 / §11). The trainer corrects the heuristic's DRAFT hit list per rally: ADD a missed
// hit, DELETE a false one, NUDGE timing. This folds the original snapshot + the edited list into
// the exact DB operations on `events` (type='hit'). Unlike rally merges, the three ops are
// independent (no consecutive-grouping); the page runs inserts → nudges → deletes so the eval
// corpus can capture the new event ids the inserts return. Mirrors ./rallyPlan.ts.

export interface HitReviewRow {
  id: string // events.id (original) or a temp id for a not-yet-saved add
  tS: number // events.t_s (seconds)
  rallyId: string | null
  rallyIdx: number | null
  confidence: number | null
  payload: unknown | null
  isNew?: boolean // true ⇒ added this session (id is a temp id, not a real events.id)
}

export interface HitSavePlan {
  inserts: Array<{ tempId: string; t_s: number; rally_id: string | null }>
  nudges: Array<{ id: string; t_s: number; orig_t_s: number }>
  deletes: Array<{ id: string; orig_t_s: number }>
  hasChanges: boolean
}

// t_s equality tolerance: events.t_s is a real (float32) column ~ms precision; anything under a
// quarter-frame @30 fps (~8 ms) is not a meaningful nudge and must not spuriously write.
const NUDGE_EPS_S = 0.008

/** German validation message for an un-saveable edited list, or null when saveable. */
export function hitValidationError(edited: HitReviewRow[]): string | null {
  for (const h of edited) {
    if (!Number.isFinite(h.tS) || h.tS < 0) return 'Ein Schlag hat eine ungültige Zeit.'
  }
  return null
}

/** Diff the original DB hit list against the edited list into events ops. Assumes
 * hitValidationError returned null. Matches originals to edits by id; isNew rows are inserts. */
export function buildHitSavePlan(original: HitReviewRow[], edited: HitReviewRow[]): HitSavePlan {
  const byOrig = new Map(original.map((o) => [o.id, o]))
  const keptIds = new Set(edited.filter((e) => !e.isNew).map((e) => e.id))

  const inserts: HitSavePlan['inserts'] = edited
    .filter((e) => e.isNew)
    .map((e) => ({ tempId: e.id, t_s: e.tS, rally_id: e.rallyId }))

  const nudges: HitSavePlan['nudges'] = []
  for (const e of edited) {
    if (e.isNew) continue
    const o = byOrig.get(e.id)
    if (o && Math.abs(e.tS - o.tS) > NUDGE_EPS_S) nudges.push({ id: e.id, t_s: e.tS, orig_t_s: o.tS })
  }

  const deletes: HitSavePlan['deletes'] = original
    .filter((o) => !keptIds.has(o.id))
    .map((o) => ({ id: o.id, orig_t_s: o.tS }))

  return {
    inserts,
    nudges,
    deletes,
    hasChanges: inserts.length > 0 || nudges.length > 0 || deletes.length > 0,
  }
}
