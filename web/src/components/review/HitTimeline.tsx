import { useRef, type PointerEvent as ReactPointerEvent } from 'react'
import { fmtTime } from '../../lib/format'
import type { HitReviewRow } from './hitPlan'

interface Props {
  hits: HitReviewRow[] // the edited working list (excludes rows marked for delete)
  dur: number
  time: number // playhead (video currentTime)
  fps: number // video fps → frame-accurate ±nudge + add step
  selectedId: string | null
  onSelect: (id: string | null) => void
  onSeek: (t: number) => void
  onAddAtPlayhead: () => void
  onNudge: (id: string, tS: number) => void
  onDelete: (id: string) => void
}

/**
 * Timeline strip of hit ticks over the video scrubber (stroke-classification-v2 §11) — the one
 * new UI primitive for the hit-editor. Click a tick = select + seek to it; drag a tick = nudge its
 * time; the toolbar adds a hit at the playhead, and deletes / frame-nudges the selected one.
 * Presentational: it owns no edit state, only emits intents (mirrors RallyReviewBoard).
 */
export function HitTimeline({
  hits, dur, time, fps, selectedId, onSelect, onSeek, onAddAtPlayhead, onNudge, onDelete,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragId = useRef<string | null>(null)
  const frame = 1 / (fps || 30)
  const pct = (t: number) => (dur > 0 ? Math.min(100, Math.max(0, (t / dur) * 100)) : 0)
  const timeAtClientX = (clientX: number): number => {
    const el = trackRef.current
    if (!el || dur <= 0) return 0
    const r = el.getBoundingClientRect()
    return Math.min(dur, Math.max(0, ((clientX - r.left) / r.width) * dur))
  }

  const onTickDown = (e: ReactPointerEvent, id: string) => {
    e.stopPropagation()
    e.currentTarget.setPointerCapture?.(e.pointerId) // optional: absent under jsdom
    dragId.current = id
    onSelect(id)
    const h = hits.find((x) => x.id === id)
    if (h) onSeek(h.tS)
  }
  const onTickMove = (e: ReactPointerEvent, id: string) => {
    if (dragId.current !== id) return
    const t = timeAtClientX(e.clientX)
    if (Number.isFinite(t)) onNudge(id, t)
  }
  const onTickUp = (e: ReactPointerEvent) => {
    if (dragId.current) {
      const h = hits.find((x) => x.id === dragId.current)
      if (h) onSeek(h.tS) // land the video on the nudged frame
    }
    dragId.current = null
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  const selected = hits.find((h) => h.id === selectedId) ?? null

  return (
    <div style={{ marginTop: 10 }}>
      <div
        ref={trackRef}
        onPointerDown={(e) => onSeek(timeAtClientX(e.clientX))} // click empty track = seek
        style={{
          position: 'relative', height: 44, borderRadius: 8, cursor: 'pointer',
          background: 'var(--hairline, hsl(214.3 31.8% 91.4%))', touchAction: 'none',
        }}
      >
        {/* centre baseline */}
        <div style={{ position: 'absolute', top: 21, left: 0, right: 0, height: 2, background: 'hsl(214.3 31.8% 85%)' }} />
        {/* hit ticks */}
        {hits.map((h) => {
          const isSel = h.id === selectedId
          const color = h.isNew ? '#1758A6' : '#FA8832' // added = opponent-blue, draft = brand-orange
          return (
            <div
              key={h.id}
              role="button"
              aria-label={`Schlag bei ${fmtTime(h.tS)}`}
              onPointerDown={(e) => onTickDown(e, h.id)}
              onPointerMove={(e) => onTickMove(e, h.id)}
              onPointerUp={onTickUp}
              style={{
                position: 'absolute', top: isSel ? 4 : 8, height: isSel ? 36 : 28,
                left: `${pct(h.tS)}%`, width: isSel ? 4 : 2, marginLeft: isSel ? -2 : -1,
                background: color, borderRadius: 2, cursor: 'ew-resize',
                boxShadow: isSel ? '0 0 0 2px rgba(26,5,0,.15)' : 'none',
              }}
            />
          )
        })}
        {/* playhead */}
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${pct(time)}%`, width: 2, marginLeft: -1, background: '#1A0500' }} />
      </div>

      {/* toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
        <button type="button" onClick={onAddAtPlayhead} style={btn}>＋ Schlag hier</button>
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>
          {hits.length} Schläge{selected ? ` · gewählt: ${fmtTime(selected.tS)}` : ''}
        </span>
        {selected && (
          <span style={{ display: 'inline-flex', gap: 6, marginLeft: 'auto' }}>
            <button type="button" aria-label="1 Frame zurück" onClick={() => onNudge(selected.id, Math.max(0, selected.tS - frame))} style={btn}>◀ 1F</button>
            <button type="button" aria-label="1 Frame vor" onClick={() => onNudge(selected.id, Math.min(dur, selected.tS + frame))} style={btn}>1F ▶</button>
            <button type="button" onClick={() => onDelete(selected.id)} style={{ ...btn, borderColor: '#C0392B', color: '#C0392B' }}>－ Löschen</button>
          </span>
        )}
      </div>
    </div>
  )
}

const btn: React.CSSProperties = {
  border: '1px solid #C66C27', background: '#FEF4EA', color: '#1A0500',
  borderRadius: 8, padding: '5px 10px', fontSize: 13, cursor: 'pointer', fontFamily: 'var(--font-body)',
}
