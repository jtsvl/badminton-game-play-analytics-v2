import type { Side } from '../../types'

// One editable mapping row = one match_players (side,slot) binding the trainer confirms/corrects.
export interface ReviewRow {
  id: string // match_players.id
  side: Side
  slot: number
  playerId: string // current player_id (registered player in this slot)
  trackLabel: number | null // current track_label (detected track id), null = none mapped yet
}

// A registered player of the match, for the player dropdown.
export interface ReviewPlayerOption {
  id: string
  name: string // "Vorname Nachname" for the option label
}

/** True when any row's player_id or track_label differs from its initial value (⇒ recompute needed). */
export function rowsChanged(initial: ReviewRow[], edited: ReviewRow[]): boolean {
  const byId = new Map(initial.map((r) => [r.id, r]))
  return edited.some((r) => {
    const o = byId.get(r.id)
    return !o || o.playerId !== r.playerId || o.trackLabel !== r.trackLabel
  })
}

/** Rows whose player_id or track_label changed vs initial (the ones Save must write). */
export function changedRows(initial: ReviewRow[], edited: ReviewRow[]): ReviewRow[] {
  const byId = new Map(initial.map((r) => [r.id, r]))
  return edited.filter((r) => {
    const o = byId.get(r.id)
    return !o || o.playerId !== r.playerId || o.trackLabel !== r.trackLabel
  })
}

/** Duplicate player_id or duplicate (non-null) track_label across rows — an invalid mapping to block. */
export function duplicateAssignments(rows: ReviewRow[]): { players: boolean; tracks: boolean } {
  const pids = rows.map((r) => r.playerId)
  const labels = rows.map((r) => r.trackLabel).filter((l): l is number => l !== null)
  return {
    players: new Set(pids).size !== pids.length,
    tracks: new Set(labels).size !== labels.length,
  }
}
