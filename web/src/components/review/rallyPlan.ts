// Rally-review save plan (FR-35 drop/merge) — pure logic, mirrors ./types.ts for PlayerReview.
// The trainer flags rows to drop and rows to merge into their predecessor; the plan lists the
// exact DB operations. Execution order (the page's job): updates → event re-parents → deletes —
// events MUST move before the absorbed rally is deleted (events.rally_id is ON DELETE SET NULL).

export interface RallyReviewRow {
  id: string // rallies.id
  idx: number
  tStart: number | null
  tEnd: number | null
  shotCount: number | null
  winnerSide: 'A' | 'B' | null
  endShotType: string | null
  confidence: number | null
  editedAt: string | null // set ⇒ trainer already edited this row
}

export interface RallyPatch {
  t_end_s: number | null
  shot_count: number | null
  winner_side: 'A' | 'B' | null
  end_shot_type: string | null
  confidence: number | null
  edited_at: string
}

export interface RallySavePlan {
  rallyUpdates: Array<{ id: string; patch: RallyPatch }>
  eventReparents: Array<{ from: string; to: string }>
  rallyDeletes: Array<{ id: string; deleteEvents: boolean }>
  hasChanges: boolean
}

/** German validation message for invalid drop/merge combinations, or null when saveable. */
export function mergeValidationError(
  rows: RallyReviewRow[],
  dropped: Set<string>,
  mergedWithPrev: Set<string>,
): string | null {
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]
    if (!mergedWithPrev.has(r.id)) continue
    if (i === 0) return 'Der erste Ballwechsel hat keinen Vorgänger zum Zusammenführen.'
    if (dropped.has(r.id) || dropped.has(rows[i - 1].id))
      return 'Zusammenführen neben einem entfernten Ballwechsel ist nicht möglich.'
  }
  return null
}

/** Fold the flags into DB operations. Assumes mergeValidationError returned null. */
export function buildRallySavePlan(
  rows: RallyReviewRow[],
  dropped: Set<string>,
  mergedWithPrev: Set<string>,
  nowIso: string,
): RallySavePlan {
  const rallyUpdates: RallySavePlan['rallyUpdates'] = []
  const eventReparents: RallySavePlan['eventReparents'] = []
  const rallyDeletes: RallySavePlan['rallyDeletes'] = []

  // Consecutive merge flags fold into one group: [keep, ...absorbed].
  let i = 0
  while (i < rows.length) {
    const keep = rows[i]
    const absorbed: RallyReviewRow[] = []
    let j = i + 1
    while (j < rows.length && mergedWithPrev.has(rows[j].id)) {
      absorbed.push(rows[j])
      j++
    }
    if (absorbed.length > 0) {
      const group = [keep, ...absorbed]
      const last = group[group.length - 1]
      const counts = group.map((r) => r.shotCount).filter((c): c is number => c !== null)
      const confs = group.map((r) => r.confidence).filter((c): c is number => c !== null)
      rallyUpdates.push({
        id: keep.id,
        patch: {
          t_end_s: last.tEnd,
          // sum is the spec-sanctioned first cut (provisional count_hits estimate anyway)
          shot_count: counts.length > 0 ? counts.reduce((a, b) => a + b, 0) : null,
          // end attributes travel with the rally end; min confidence — the merged span
          // includes the dropout gap, so the weaker estimate is the honest one
          winner_side: last.winnerSide,
          end_shot_type: last.endShotType,
          confidence: confs.length > 0 ? Math.min(...confs) : null,
          edited_at: nowIso,
        },
      })
      for (const r of absorbed) {
        eventReparents.push({ from: r.id, to: keep.id })
        rallyDeletes.push({ id: r.id, deleteEvents: false }) // events already re-parented
      }
    }
    i = j
  }

  for (const r of rows) {
    if (dropped.has(r.id)) rallyDeletes.push({ id: r.id, deleteEvents: true })
  }

  return {
    rallyUpdates,
    eventReparents,
    rallyDeletes,
    hasChanges: rallyUpdates.length > 0 || rallyDeletes.length > 0,
  }
}
