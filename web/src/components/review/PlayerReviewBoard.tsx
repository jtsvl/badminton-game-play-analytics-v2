import { useMemo, useState } from 'react'
import type { Discipline } from '../../types'
import { type TrackKey } from '../../lib/overlay'
import { PlayerReviewOverlay } from './PlayerReviewOverlay'
import {
  type ReviewPlayerOption,
  type ReviewRow,
  changedRows,
  duplicateAssignments,
  rowsChanged,
} from './types'

interface Props {
  initialRows: ReviewRow[]
  players: ReviewPlayerOption[] // registered players of this match (player dropdown)
  pathByLabel: Map<number, TrackKey[]> // detected track id → keyframes (for the overlay)
  H: number[] | null
  frameUrl: string | null
  size: { w: number; h: number }
  discipline: Discipline
  saving: boolean
  saveError: string | null
  onConfirm: () => void // no change → mark match done
  onSave: (edited: ReviewRow[]) => void // change → write rows + enqueue recompute
}

const ORANGE = '#FA8832'
const BLUE = '#1758A6'
const sideColor = (side: ReviewRow['side']) => (side === 'A' ? ORANGE : BLUE)
const sideLabel = (side: ReviewRow['side']) => (side === 'A' ? 'Feld A' : 'Feld B')

/**
 * Presentational player-identity review board (no data layer — testable with a fixture). The trainer
 * reassigns the registered player and/or the detected track per (side,slot), sees each track's path on
 * the still, then confirms (no change → done) or saves (change → recompute). Mirrors the Calibrate
 * sidebar/controls shape; German, light mode, brand tokens.
 */
export function PlayerReviewBoard({
  initialRows,
  players,
  pathByLabel,
  H,
  frameUrl,
  size,
  discipline,
  saving,
  saveError,
  onConfirm,
  onSave,
}: Props) {
  const [edited, setEdited] = useState<ReviewRow[]>(initialRows)
  const [focused, setFocused] = useState<number | null>(null)

  // Detected tracks the trainer can pick from = the labels the worker emitted across the slots.
  const availableLabels = useMemo(
    () => Array.from(new Set(initialRows.map((r) => r.trackLabel).filter((l): l is number => l !== null))).sort((a, b) => a - b),
    [initialRows],
  )
  const nameById = useMemo(() => new Map(players.map((p) => [p.id, p.name])), [players])

  const changed = rowsChanged(initialRows, edited)
  const dup = duplicateAssignments(edited)
  const hasDup = dup.players || dup.tracks
  const canSave = changed && !hasDup && !saving
  const canConfirm = !changed && !saving

  const setRow = (i: number, patch: Partial<ReviewRow>) =>
    setEdited((rows) => rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr)',
        gap: 20,
      }}
    >
      <PlayerReviewOverlay H={H} frameUrl={frameUrl} size={size} rows={edited} pathByLabel={pathByLabel} focused={focused} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
          Ordne jeder Position den richtigen Spieler und die erkannte Spur (Laufweg) zu. Die Zuordnung ist
          eine <strong>Schätzung</strong> – bitte prüfen und ggf. korrigieren.
        </p>

        {edited.map((row, i) => (
          <div
            key={row.id}
            onMouseEnter={() => setFocused(i)}
            onMouseLeave={() => setFocused(null)}
            style={{
              border: `1px solid ${focused === i ? sideColor(row.side) : 'var(--hairline, hsl(214.3 31.8% 91.4%))'}`,
              borderLeft: `4px solid ${sideColor(row.side)}`,
              borderRadius: 12,
              padding: '12px 14px',
              background: 'var(--card, #fff)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <span style={{ fontFamily: 'var(--font-head, Space Grotesk)', fontWeight: 700, fontSize: 14 }}>
                {sideLabel(row.side)}
                {discipline === 'doubles' ? ` · Pos. ${row.slot + 1}` : ''}
              </span>
              <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                {row.trackLabel !== null ? `Spur #${row.trackLabel}` : 'keine Spur'}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                Spieler
                <select
                  value={row.playerId}
                  onChange={(e) => setRow(i, { playerId: e.target.value })}
                  style={selectStyle}
                >
                  {players.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                  {!nameById.has(row.playerId) && <option value={row.playerId}>Unbekannt</option>}
                </select>
              </label>

              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                Erkannte Spur
                <select
                  value={row.trackLabel === null ? '' : String(row.trackLabel)}
                  onChange={(e) => setRow(i, { trackLabel: e.target.value === '' ? null : Number(e.target.value) })}
                  style={selectStyle}
                >
                  <option value="">— keine —</option>
                  {availableLabels.map((l) => (
                    <option key={l} value={String(l)}>
                      Spur #{l}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        ))}

        {hasDup && (
          <p style={{ margin: 0, fontSize: 12.5, color: 'var(--heat-max, #C0392B)' }}>
            {dup.players && 'Ein Spieler ist mehrfach zugeordnet. '}
            {dup.tracks && 'Eine Spur ist mehrfach zugeordnet. '}
            Bitte eindeutig zuordnen, bevor du speicherst.
          </p>
        )}

        {saveError && <p style={{ margin: 0, fontSize: 12.5, color: 'var(--heat-max, #C0392B)' }}>{saveError}</p>}

        <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
          <button type="button" onClick={onConfirm} disabled={!canConfirm} style={btnStyle(canConfirm, false)}>
            Bestätigen
          </button>
          <button type="button" onClick={() => onSave(changedRows(initialRows, edited))} disabled={!canSave} style={btnStyle(canSave, true)}>
            {saving ? 'Speichert…' : 'Speichern & neu berechnen'}
          </button>
        </div>
        <p style={{ margin: 0, fontSize: 11.5, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
          {changed
            ? 'Speichern berechnet Statistiken und Overlays neu.'
            : 'Bestätigen schließt das Spiel ohne Neuberechnung ab.'}
        </p>
      </div>
    </div>
  )
}

const selectStyle: React.CSSProperties = {
  padding: '8px 10px',
  borderRadius: 8,
  border: '1px solid var(--hairline, hsl(214.3 31.8% 91.4%))',
  background: '#fff',
  color: 'var(--ink, #0A0A0B)',
  fontSize: 14,
  fontFamily: 'var(--font-body, Inter)',
}

const btnStyle = (enabled: boolean, primary: boolean): React.CSSProperties => ({
  flex: 1,
  padding: '10px 14px',
  borderRadius: 10,
  border: primary ? 'none' : '1px solid var(--hairline, hsl(214.3 31.8% 91.4%))',
  background: primary ? (enabled ? 'var(--primary, #FA8832)' : 'var(--hairline, hsl(214.3 31.8% 91.4%))') : '#fff',
  color: primary ? (enabled ? 'var(--dark-on-orange, #1A0500)' : 'var(--muted, hsl(215.4 16.3% 46.9%))') : 'var(--ink, #0A0A0B)',
  fontWeight: 700,
  fontSize: 14,
  cursor: enabled ? 'pointer' : 'not-allowed',
  fontFamily: 'var(--font-body, Inter)',
})
