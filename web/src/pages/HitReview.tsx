import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { supabase, isConfigured } from '../lib/supabase'
import { resolvePlaybackSrc } from '../lib/storage'
import { useVideoPlayer } from '../hooks/useVideoPlayer'
import { VideoControls } from '../components/report/VideoControls'
import { HitTimeline } from '../components/review/HitTimeline'
import { buildHitSavePlan, hitValidationError, type HitReviewRow } from '../components/review/hitPlan'
import { buildHitCorrections } from '../components/review/hitCorrections'

interface RallySpan { id: string; idx: number; tStart: number | null; tEnd: number | null }
interface Ctx {
  matchId: string
  videoId: string
  status: string | null
  fps: number
  original: HitReviewRow[]
  rallies: RallySpan[]
  videoSrc: string
}

/**
 * /match/:slug/hits — correct the heuristic's DRAFT hit list (add missed / delete false / nudge
 * timing) into dense, unbiased ground truth for the learned hit-detector (stroke-classification-v2
 * §5 Stage 1 / §11). Mirrors RallyReview: thin container, presentational timeline, diff-only writes
 * with an assertRows guard (PostgREST reports RLS-blocked writes as success-with-zero-rows). The
 * corrected hit list rides a `hit_review` job as params.corrections → hit_corrections (0013).
 */
export function HitReview() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [ctx, setCtx] = useState<Ctx | null>(null)
  const [edited, setEdited] = useState<HitReviewRow[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const hasFootage = Boolean(ctx?.videoSrc)
  const player = useVideoPlayer(slug ? slug + '/hits' : undefined, hasFootage)

  useEffect(() => {
    if (!slug) return
    if (!isConfigured) {
      setPhase('error'); setErrorMsg('Supabase ist nicht konfiguriert (.env).'); return
    }
    let cancelled = false
    ;(async () => {
      try {
        const { data: match } = await supabase.from('matches').select('id, status').eq('slug', slug).single()
        if (!match) throw new Error('Spiel nicht gefunden')
        const { data: video } = await supabase
          .from('match_videos').select('id, storage_path, proxy_path, fps')
          .eq('match_id', match.id).limit(1).maybeSingle()
        if (!video) throw new Error('Kein Video für dieses Spiel')
        const videoSrc = await resolvePlaybackSrc(video as { storage_path: string; proxy_path: string | null })

        const { data: rallies } = await supabase
          .from('rallies').select('id, idx, t_start_s, t_end_s').eq('match_id', match.id).order('idx')
        const rSpans: RallySpan[] = (rallies ?? []).map((r) => ({ id: r.id, idx: r.idx, tStart: r.t_start_s, tEnd: r.t_end_s }))
        const idxById = new Map(rSpans.map((r) => [r.id, r.idx]))

        const { data: events, error: eErr } = await supabase
          .from('events').select('id, t_s, rally_id, confidence, payload')
          .eq('match_id', match.id).eq('type', 'hit').order('t_s')
        if (eErr) throw eErr
        const original: HitReviewRow[] = (events ?? []).map((e) => ({
          id: e.id, tS: e.t_s, rallyId: e.rally_id,
          rallyIdx: e.rally_id ? idxById.get(e.rally_id) ?? null : null,
          confidence: e.confidence, payload: e.payload, isNew: false,
        }))

        if (cancelled) return
        setCtx({
          matchId: match.id, videoId: video.id, status: match.status ?? null,
          fps: Number(video.fps) || 30, original, rallies: rSpans, videoSrc,
        })
        setEdited(original)
        setPhase('ready')
      } catch (e: unknown) {
        if (cancelled) return
        setErrorMsg((e instanceof Error ? e.message : null) ?? 'Fehler beim Laden')
        setPhase('error')
      }
    })()
    return () => { cancelled = true }
  }, [slug])

  const rallyAt = (t: number): RallySpan | null =>
    ctx?.rallies.find((r) => r.tStart !== null && r.tEnd !== null && t >= r.tStart && t <= r.tEnd) ?? null

  const onAddAtPlayhead = () => {
    const t = player.time
    const r = rallyAt(t)
    const id = 'tmp-' + crypto.randomUUID()
    setEdited((prev) => [...prev, { id, tS: t, rallyId: r?.id ?? null, rallyIdx: r?.idx ?? null, confidence: 1, payload: null, isNew: true }])
    setSelectedId(id)
  }
  const onNudge = (id: string, tS: number) => setEdited((prev) => prev.map((h) => (h.id === id ? { ...h, tS } : h)))
  const onDelete = (id: string) => { setEdited((prev) => prev.filter((h) => h.id !== id)); setSelectedId(null) }

  const onSave = async () => {
    if (!ctx) return
    const invalid = hitValidationError(edited)
    if (invalid) { setSaveError(invalid); return }
    setSaving(true); setSaveError(null)
    try {
      const assertRows = (data: unknown[] | null, what: string) => {
        if (!data || data.length === 0) throw new Error(`Speichern fehlgeschlagen — ${what} hat keine Zeile geändert (RLS?).`)
      }
      const plan = buildHitSavePlan(ctx.original, edited)
      const insertedIds = new Map<string, string>()
      for (const ins of plan.inserts) {
        const res = await supabase.from('events')
          .insert({ match_id: ctx.matchId, rally_id: ins.rally_id, t_s: ins.t_s, type: 'hit', confidence: 1 })
          .select('id').single()
        if (res.error) throw res.error
        insertedIds.set(ins.tempId, res.data.id)
      }
      for (const n of plan.nudges) {
        const res = await supabase.from('events').update({ t_s: n.t_s }).eq('id', n.id).select('id')
        if (res.error) throw res.error
        assertRows(res.data, 'Zeit ändern')
      }
      for (const d of plan.deletes) {
        const res = await supabase.from('events').delete().eq('id', d.id).select('id')
        if (res.error) throw res.error
        assertRows(res.data, 'Löschen')
      }

      const corrections = buildHitCorrections(ctx.original, edited, plan, insertedIds)
      const jb = await supabase.from('processing_jobs').insert({
        match_id: ctx.matchId, video_id: ctx.videoId, stage: 'hit_review', status: 'queued',
        params: { return_status: 'unchanged', ...(corrections.length > 0 ? { corrections } : {}) },
      })
      if (jb.error) throw jb.error
      navigate(`/match/${slug}`)
    } catch (e: unknown) {
      setSaveError((e instanceof Error ? e.message : null) ?? 'Speichern fehlgeschlagen')
      setSaving(false)
    }
  }

  if (phase === 'loading') return <Centered>Lädt…</Centered>
  if (phase === 'error' || !ctx) return <Centered>{errorMsg ?? 'Spiel nicht gefunden.'}</Centered>

  const changes = buildHitSavePlan(ctx.original, edited).hasChanges

  return (
    <main style={{ minHeight: '100%', background: 'var(--beige, #FEF4EA)', color: 'var(--ink, #0A0A0B)', fontFamily: 'var(--font-body)' }}>
      <div style={{ maxWidth: 900, margin: '0 auto', padding: 16 }}>
        <button type="button" onClick={() => navigate(`/match/${slug}`)}
          style={{ border: 'none', background: 'none', color: 'var(--muted)', fontSize: 14, cursor: 'pointer', padding: 0, marginBottom: 8 }}>
          ‹ Zurück zum Bericht
        </button>
        <h1 style={{ fontFamily: 'var(--font-head)', fontSize: 24, margin: '0 0 4px' }}>Schläge prüfen</h1>
        <p style={{ margin: '0 0 14px', fontSize: 13.5, color: 'var(--muted)' }}>
          Fehlenden Schlag hinzufügen (＋), falschen entfernen (Tick wählen → Löschen), Zeit per Ziehen oder ◀/▶ 1 Frame justieren.
        </p>

        {hasFootage && (
          <div style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--beige, #FEF4EA)', paddingBottom: 10 }}>
            <div style={{ borderRadius: 10, overflow: 'hidden', background: '#0A0A0B', lineHeight: 0 }}>
              <video ref={player.vref} src={ctx.videoSrc} playsInline preload="metadata"
                onLoadedMetadata={player.onLoaded} onEnded={() => player.setPlaying(false)}
                onTimeUpdate={() => { const v = player.vref.current; if (v && v.paused) player.setTime(v.currentTime) }}
                style={{ width: '100%', display: 'block' }} />
            </div>
            <VideoControls player={player} style={{ marginTop: 10 }} showRate fps={ctx.fps} />
            <HitTimeline
              hits={edited} dur={player.dur} time={player.time} fps={ctx.fps} selectedId={selectedId}
              onSelect={setSelectedId} onSeek={player.seek} onAddAtPlayhead={onAddAtPlayhead}
              onNudge={onNudge} onDelete={onDelete}
            />
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16 }}>
          <button type="button" disabled={!changes || saving} onClick={onSave}
            style={{
              border: '1px solid #C66C27', borderRadius: 10, padding: '10px 18px', fontSize: 15,
              cursor: changes && !saving ? 'pointer' : 'not-allowed', background: changes ? '#FA8832' : '#eadfd3',
              color: '#1A0500', boxShadow: changes ? '0 3px 0 #C66C27' : 'none', fontFamily: 'var(--font-body)',
            }}>
            {saving ? 'Speichert…' : 'Speichern'}
          </button>
          {saveError && <span style={{ color: '#C0392B', fontSize: 13 }}>{saveError}</span>}
        </div>
      </div>
    </main>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
}
