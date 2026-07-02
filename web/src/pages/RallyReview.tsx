import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { supabase, isConfigured } from '../lib/supabase'
import { resolvePlaybackSrc } from '../lib/storage'
import { useMediaQuery } from '../hooks/useMediaQuery'
import { useVideoPlayer } from '../hooks/useVideoPlayer'
import { VideoControls } from '../components/report/VideoControls'
import { RallyReviewBoard } from '../components/review/RallyReviewBoard'
import type { RallyReviewRow, RallySavePlan } from '../components/review/rallyPlan'
import type { RallyRow } from '../types'

interface Ctx {
  matchId: string
  videoId: string
  status: string | null
  rows: RallyReviewRow[]
  videoSrc: string // 720p proxy signed URL ('' → list-only fallback, no player)
}

/**
 * /match/:slug/rallies — drop false rallies / merge split rallies (FR-35), then enqueue the cheap
 * `rally_review` recompute. Mirrors PlayerReview: thin container, presentational board, diff-only
 * writes. Every update/delete asserts its affected row count — PostgREST reports RLS-blocked writes
 * as success-with-zero-rows, which must not look like a saved edit.
 */
export function RallyReview() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const isDesktop = useMediaQuery('(min-width: 1200px)')
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [ctx, setCtx] = useState<Ctx | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  // Distinct position key (slug + '/rallies') — the report page persists under 'bca-vt-'+slug.
  const hasFootage = Boolean(ctx?.videoSrc)
  const player = useVideoPlayer(slug ? slug + '/rallies' : undefined, hasFootage)

  useEffect(() => {
    if (!slug) return
    if (!isConfigured) {
      setPhase('error')
      setErrorMsg('Supabase ist nicht konfiguriert (.env).')
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const { data: match } = await supabase
          .from('matches')
          .select('id, status')
          .eq('slug', slug)
          .single()
        if (!match) throw new Error('Spiel nicht gefunden')

        const { data: video } = await supabase
          .from('match_videos')
          .select('id, storage_path, proxy_path')
          .eq('match_id', match.id)
          .limit(1)
          .maybeSingle()
        if (!video) throw new Error('Kein Video für dieses Spiel')
        const videoSrc = await resolvePlaybackSrc(
          video as { storage_path: string; proxy_path: string | null },
        )

        const { data: rallies, error: rErr } = await supabase
          .from('rallies')
          .select('id, idx, t_start_s, t_end_s, shot_count, winner_side, end_shot_type, confidence, edited_at')
          .eq('match_id', match.id)
          .order('idx')
        if (rErr) throw rErr

        const rows: RallyReviewRow[] = ((rallies ?? []) as RallyRow[]).map((r) => ({
          id: r.id,
          idx: r.idx,
          tStart: r.t_start_s,
          tEnd: r.t_end_s,
          shotCount: r.shot_count,
          winnerSide: r.winner_side,
          endShotType: r.end_shot_type,
          confidence: r.confidence,
          editedAt: r.edited_at,
        }))
        if (rows.length === 0) throw new Error('Keine Ballwechsel für dieses Spiel erkannt.')

        if (cancelled) return
        setCtx({ matchId: match.id, videoId: video.id, status: match.status ?? null, rows, videoSrc })
        setPhase('ready')
      } catch (e: unknown) {
        if (cancelled) return
        const message = e instanceof Error ? e.message : (e as { message?: string } | null)?.message
        setErrorMsg(message ?? 'Fehler beim Laden')
        setPhase('error')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [slug])

  const onSave = async (plan: RallySavePlan) => {
    if (!ctx) return
    setSaving(true)
    setSaveError(null)
    try {
      const assertRows = (data: unknown[] | null, what: string) => {
        if (!data || data.length === 0)
          throw new Error(`Speichern fehlgeschlagen — ${what} hat keine Zeile geändert (RLS?).`)
      }

      // Order is load-bearing: merge updates → event re-parents → deletes.
      // events.rally_id is ON DELETE SET NULL — events must move BEFORE their old rally is deleted.
      for (const u of plan.rallyUpdates) {
        const res = await supabase.from('rallies').update(u.patch).eq('id', u.id).select('id')
        if (res.error) throw res.error
        assertRows(res.data, 'Zusammenführen')
      }
      for (const rp of plan.eventReparents) {
        // A rally may legitimately have zero events — don't assert a row count here.
        const res = await supabase.from('events').update({ rally_id: rp.to }).eq('rally_id', rp.from)
        if (res.error) throw res.error
      }
      for (const d of plan.rallyDeletes) {
        if (d.deleteEvents) {
          const ev = await supabase.from('events').delete().eq('rally_id', d.id)
          if (ev.error) throw ev.error
        }
        const res = await supabase.from('rallies').delete().eq('id', d.id).select('id')
        if (res.error) throw res.error
        assertRows(res.data, 'Entfernen')
      }

      const now = new Date().toISOString()
      const ms = await supabase
        .from('matches')
        .update({ status: 'processing', rally_edited_at: now })
        .eq('id', ctx.matchId)
        .select('id')
      if (ms.error) throw ms.error
      assertRows(ms.data, 'Status setzen')

      // Cheap recompute: remap → rally_sync → aggregate. return_status brings the match back to
      // where it was — editing at 'review' must not skip the player confirmation.
      const jb = await supabase.from('processing_jobs').insert({
        match_id: ctx.matchId,
        video_id: ctx.videoId,
        stage: 'rally_review',
        status: 'queued',
        params: { return_status: ctx.status === 'done' ? 'done' : 'review' },
      })
      if (jb.error) throw jb.error
      navigate(`/match/${slug}`)
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : (e as { message?: string } | null)?.message
      setSaveError(message ?? 'Speichern fehlgeschlagen')
      setSaving(false)
    }
  }

  if (phase === 'loading') return <Centered>Lädt…</Centered>
  if (phase === 'error' || !ctx) return <Centered>{errorMsg ?? 'Spiel nicht gefunden.'}</Centered>

  // Seek to just before the rally start and play. Deliberately NO auto-stop at the rally end:
  // for a dropout split, play visibly continuing past the detected end is the merge evidence.
  const onPlayRally = (row: RallyReviewRow) => {
    player.seek(Math.max(0, (row.tStart ?? 0) - 1))
    if (!player.playing) player.togglePlay()
  }
  const activeRallyId =
    ctx.rows.find(
      (r) => r.tStart !== null && r.tEnd !== null && player.time >= r.tStart && player.time <= r.tEnd,
    )?.id ?? null

  return (
    <main style={{ minHeight: '100%', background: 'var(--beige, #FEF4EA)', color: 'var(--ink, #0A0A0B)', fontFamily: 'var(--font-body)' }}>
      <div style={{ maxWidth: 760, margin: '0 auto', padding: isDesktop ? 24 : 14 }}>
        <button
          type="button"
          onClick={() => navigate(`/match/${slug}`)}
          style={{ border: 'none', background: 'none', color: 'var(--muted)', fontSize: 14, cursor: 'pointer', padding: 0, marginBottom: 8 }}
        >
          ‹ Zurück zum Bericht
        </button>
        <h1 style={{ fontFamily: 'var(--font-head)', fontSize: isDesktop ? 28 : 22, margin: '0 0 16px' }}>Ballwechsel prüfen</h1>

        {hasFootage && (
          <div
            style={{
              position: 'sticky',
              top: 0,
              zIndex: 2,
              background: 'var(--beige, #FEF4EA)',
              paddingBottom: 10,
              marginBottom: 6,
            }}
          >
            <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#0A0A0B', lineHeight: 0 }}>
              {/* Not muted — the hit sounds help judge rally boundaries. */}
              <video
                ref={player.vref}
                src={ctx.videoSrc}
                playsInline
                preload="metadata"
                onLoadedMetadata={player.onLoaded}
                onEnded={() => player.setPlaying(false)}
                onTimeUpdate={() => {
                  const v = player.vref.current
                  if (v && v.paused) player.setTime(v.currentTime)
                }}
                style={{ width: '100%', display: 'block' }}
              />
            </div>
            <VideoControls player={player} style={{ marginTop: 10 }} />
          </div>
        )}

        <RallyReviewBoard
          initialRows={ctx.rows}
          saving={saving}
          saveError={saveError}
          onSave={onSave}
          onPlayRally={hasFootage ? onPlayRally : undefined}
          activeRallyId={hasFootage ? activeRallyId : undefined}
        />
      </div>
    </main>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
}
