import { useState } from 'react'
import { deNum, fmtTime } from '../../lib/format'
import {
  type RallyReviewRow,
  type RallySavePlan,
  buildRallySavePlan,
  mergeValidationError,
} from './rallyPlan'

interface Props {
  initialRows: RallyReviewRow[] // ordered by idx
  saving: boolean
  saveError: string | null
  onSave: (plan: RallySavePlan) => void // flags → DB operations + recompute job
  onPlayRally?: (row: RallyReviewRow) => void // container seeks its video player (video-proxy)
  activeRallyId?: string | null // row under the playhead → highlighted
}

/**
 * Presentational rally-review board (no data layer — testable with a fixture). The trainer removes
 * false rallies (neighbour-court leaks) and merges rallies a dropout split; Save re-gates the stats.
 * Mirrors PlayerReviewBoard: list-first, German du-form, brand tokens. The video player lives in the
 * container (RallyReview) — the board only exposes a per-row „Ansehen" seek affordance.
 */
export function RallyReviewBoard({ initialRows, saving, saveError, onSave, onPlayRally, activeRallyId }: Props) {
  const [dropped, setDropped] = useState<Set<string>>(new Set())
  const [merged, setMerged] = useState<Set<string>>(new Set())

  const toggle = (set: Set<string>, id: string) => {
    const next = new Set(set)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  }

  const validation = mergeValidationError(initialRows, dropped, merged)
  const hasFlags = dropped.size > 0 || merged.size > 0
  const canSave = hasFlags && !validation && !saving

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <p style={{ margin: 0, fontSize: 13, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
        Die Ballwechsel sind eine <strong>Schätzung</strong> aus dem Video. Entferne falsche Ballwechsel
        (z.&nbsp;B. vom Nachbarfeld) und führe getrennte Teile eines Ballwechsels zusammen.
      </p>

      {initialRows.map((r, i) => {
        const isDropped = dropped.has(r.id)
        const isMerged = merged.has(r.id)
        const isActive = activeRallyId === r.id
        const dur = r.tStart !== null && r.tEnd !== null ? r.tEnd - r.tStart : null
        return (
          <div key={r.id} style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {i > 0 && (
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  color: isMerged ? 'var(--ink, #0A0A0B)' : 'var(--muted, hsl(215.4 16.3% 46.9%))',
                  padding: '2px 14px 6px',
                }}
              >
                <input
                  type="checkbox"
                  checked={isMerged}
                  onChange={() => setMerged((s) => toggle(s, r.id))}
                  aria-label={`Ballwechsel ${i + 1} mit vorherigem zusammenführen`}
                />
                Mit vorherigem zusammenführen
              </label>
            )}
            <div
              id={`rally-${r.id}`}
              aria-current={isActive || undefined}
              style={{
                // Per-side borders (no `border` shorthand): the active highlight rerenders these
                // values during playback, and React warns when a shorthand and a longhand mix.
                borderTop: `1px solid ${isActive ? 'var(--primary, #FA8832)' : 'var(--hairline, hsl(214.3 31.8% 91.4%))'}`,
                borderRight: `1px solid ${isActive ? 'var(--primary, #FA8832)' : 'var(--hairline, hsl(214.3 31.8% 91.4%))'}`,
                borderBottom: `1px solid ${isActive ? 'var(--primary, #FA8832)' : 'var(--hairline, hsl(214.3 31.8% 91.4%))'}`,
                borderLeft: `4px solid ${isDropped ? 'var(--heat-max, #C0392B)' : 'var(--primary, #FA8832)'}`,
                borderRadius: 12,
                padding: '12px 14px',
                background: isActive ? 'rgba(250,136,50,.06)' : 'var(--card, #fff)',
                opacity: isDropped ? 0.55 : 1,
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                flexWrap: 'wrap',
              }}
            >
              {onPlayRally && (
                <button
                  type="button"
                  onClick={() => onPlayRally(r)}
                  aria-label={`Ballwechsel ${i + 1} ansehen`}
                  title="Ansehen"
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 14,
                    border: '1px solid var(--orange-shade, #C66C27)',
                    background: 'var(--primary, #FA8832)',
                    color: 'var(--dark-on-orange, #1A0500)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 0,
                    flex: 'none',
                  }}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M7 4.8v14.4c0 .9 1 1.5 1.8 1L20 13c.8-.5.8-1.6 0-2.1L8.8 3.9C8 3.4 7 4 7 4.8z" />
                  </svg>
                </button>
              )}
              <span style={{ fontFamily: 'var(--font-head, Space Grotesk)', fontWeight: 700, fontSize: 14, minWidth: 88 }}>
                Ballwechsel {i + 1}
              </span>
              <span className="tabular" style={{ fontSize: 13 }}>
                {r.tStart !== null && r.tEnd !== null ? `${fmtTime(r.tStart)} – ${fmtTime(r.tEnd)}` : '–'}
              </span>
              <span style={{ fontSize: 12.5, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                {dur !== null ? `Dauer ${fmtTime(dur)}` : ''}
              </span>
              <span style={{ fontSize: 12.5, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                {r.shotCount !== null ? `${deNum(r.shotCount)} Schläge` : 'Schläge unbekannt'}
              </span>
              <span style={{ fontSize: 12.5, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
                {r.confidence !== null ? `Konfidenz ${deNum(r.confidence * 100)} %` : ''}
              </span>
              {r.editedAt && (
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    background: 'rgba(250,136,50,.16)',
                    color: 'var(--orange-shade, #C66C27)',
                    borderRadius: 99,
                    padding: '2px 8px',
                  }}
                >
                  bearbeitet
                </span>
              )}
              <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5 }}>
                <input
                  type="checkbox"
                  checked={isDropped}
                  onChange={() => setDropped((s) => toggle(s, r.id))}
                  aria-label={`Ballwechsel ${i + 1} entfernen`}
                />
                Entfernen
              </label>
            </div>
          </div>
        )
      })}

      {validation && <p style={{ margin: 0, fontSize: 12.5, color: 'var(--heat-max, #C0392B)' }}>{validation}</p>}
      {saveError && <p style={{ margin: 0, fontSize: 12.5, color: 'var(--heat-max, #C0392B)' }}>{saveError}</p>}

      <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
        <button
          type="button"
          onClick={() => onSave(buildRallySavePlan(initialRows, dropped, merged, new Date().toISOString()))}
          disabled={!canSave}
          style={btnStyle(canSave)}
        >
          {saving ? 'Speichert…' : 'Speichern & neu berechnen'}
        </button>
      </div>
      <p style={{ margin: 0, fontSize: 11.5, color: 'var(--muted, hsl(215.4 16.3% 46.9%))' }}>
        Speichern übernimmt deine Änderungen und berechnet die Statistiken neu.
      </p>
    </div>
  )
}

const btnStyle = (enabled: boolean): React.CSSProperties => ({
  flex: 1,
  padding: '10px 14px',
  borderRadius: 10,
  border: 'none',
  background: enabled ? 'var(--primary, #FA8832)' : 'var(--hairline, hsl(214.3 31.8% 91.4%))',
  color: enabled ? 'var(--dark-on-orange, #1A0500)' : 'var(--muted, hsl(215.4 16.3% 46.9%))',
  fontWeight: 700,
  fontSize: 14,
  cursor: enabled ? 'pointer' : 'not-allowed',
  fontFamily: 'var(--font-body, Inter)',
})
