import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { supabase, isConfigured } from '../lib/supabase'
import { signedUrl } from '../lib/storage'
import { homographyFromCorners } from '../lib/calibration'
import type { Pt } from '../lib/homography'
import type { TrackKey } from '../lib/overlay'
import type { Discipline, Side } from '../types'
import { useMediaQuery } from '../hooks/useMediaQuery'
import { PlayerReviewBoard } from '../components/review/PlayerReviewBoard'
import type { ReviewPlayerOption, ReviewRow } from '../components/review/types'

interface Ctx {
  matchId: string
  videoId: string
  discipline: Discipline
  status: string | null
  rows: ReviewRow[]
  players: ReviewPlayerOption[]
  pathByLabel: Map<number, TrackKey[]>
  H: number[] | null
  frameUrl: string | null
  size: { w: number; h: number }
}

function safeH(corners: Pt[] | null, size: { w: number; h: number } | null): number[] | null {
  if (!corners || corners.length !== 4 || !size) return null
  try {
    return homographyFromCorners(corners, size)
  } catch {
    return null
  }
}

/** /match/:slug/review — confirm or correct the auto-estimated track→player mapping, then finish. */
export function PlayerReview() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const isDesktop = useMediaQuery('(min-width: 1200px)')
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [ctx, setCtx] = useState<Ctx | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

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
          .select('id, status, discipline')
          .eq('slug', slug)
          .single()
        if (!match) throw new Error('Spiel nicht gefunden')

        const { data: video } = await supabase
          .from('match_videos')
          .select('id, storage_path, width, height')
          .eq('match_id', match.id)
          .limit(1)
          .maybeSingle()
        if (!video) throw new Error('Kein Video für dieses Spiel')

        const [calibRes, mpRes, tkRes] = await Promise.all([
          supabase
            .from('court_calibrations')
            .select('confirmed_points, image_size, frame_uri')
            .eq('video_id', video.id)
            .order('updated_at', { ascending: false })
            .limit(1)
            .maybeSingle(),
          supabase
            .from('match_players')
            .select('id, side, slot, player_id, track_label, players(name, first_name)')
            .eq('match_id', match.id),
          supabase
            .from('track_keyframes')
            .select('subject, data')
            .eq('video_id', video.id)
            .like('subject', 'player:%'),
        ])

        const calib = calibRes.data as
          | { confirmed_points: Pt[] | null; image_size: { w: number; h: number } | null; frame_uri: string | null }
          | null
        const mps = (mpRes.data ?? []) as Array<{
          id: string
          side: Side
          slot: number
          player_id: string
          track_label: number | null
          players: { name: string; first_name: string } | { name: string; first_name: string }[] | null
        }>
        if (mps.length === 0) throw new Error('Keine Spieler für dieses Spiel')

        const size = (calib?.image_size as { w: number; h: number } | null) ?? {
          w: video.width ?? 1920,
          h: video.height ?? 1080,
        }
        const H = safeH((calib?.confirmed_points as Pt[] | null) ?? null, size)
        const frameUrl =
          (calib?.frame_uri ? await signedUrl('analysis-artifacts', calib.frame_uri) : '') ||
          (await signedUrl('match-videos', video.storage_path)) ||
          null

        // Sort rows for a stable layout (Feld A first, then by slot).
        const sorted = [...mps].sort((a, b) => (a.side === b.side ? a.slot - b.slot : a.side < b.side ? -1 : 1))
        const rows: ReviewRow[] = sorted.map((mp) => ({
          id: mp.id,
          side: mp.side,
          slot: mp.slot,
          playerId: mp.player_id,
          trackLabel: mp.track_label ?? null,
        }))
        const players: ReviewPlayerOption[] = sorted.map((mp) => {
          const p = Array.isArray(mp.players) ? mp.players[0] : mp.players
          return { id: mp.player_id, name: p?.name ?? 'Unbekannt' } // players.name is the full name
        })

        // Each player's keyframes, keyed by subject; a track's path = the keys of the player it is
        // currently bound to (overlay wrote 'player:<id>' from that track_label's rows).
        const playerKeys = new Map<string, TrackKey[]>(
          ((tkRes.data ?? []) as Array<{ subject: string; data: TrackKey[] | null }>).map((t) => [t.subject, t.data ?? []]),
        )
        const pathByLabel = new Map<number, TrackKey[]>()
        for (const mp of sorted) {
          if (mp.track_label !== null && mp.track_label !== undefined) {
            pathByLabel.set(mp.track_label, playerKeys.get(`player:${mp.player_id}`) ?? [])
          }
        }

        if (cancelled) return
        setCtx({
          matchId: match.id,
          videoId: video.id,
          discipline: (match.discipline as Discipline) ?? 'singles',
          status: match.status ?? null,
          rows,
          players,
          pathByLabel,
          H,
          frameUrl,
          size,
        })
        setPhase('ready')
      } catch (e: unknown) {
        if (cancelled) return
        setErrorMsg(e instanceof Error ? e.message : 'Fehler beim Laden')
        setPhase('error')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [slug])

  const onConfirm = async () => {
    if (!ctx) return
    setSaving(true)
    setSaveError(null)
    try {
      const ms = await supabase.from('matches').update({ status: 'done' }).eq('id', ctx.matchId)
      if (ms.error) throw ms.error
      navigate(`/match/${slug}`)
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : 'Speichern fehlgeschlagen')
      setSaving(false)
    }
  }

  const onSave = async (changed: ReviewRow[]) => {
    if (!ctx) return
    setSaving(true)
    setSaveError(null)
    try {
      for (const r of changed) {
        const up = await supabase
          .from('match_players')
          .update({ player_id: r.playerId, track_label: r.trackLabel })
          .eq('id', r.id)
        if (up.error) throw up.error
      }
      const ms = await supabase.from('matches').update({ status: 'processing' }).eq('id', ctx.matchId)
      if (ms.error) throw ms.error
      // Cheap recompute: remap → aggregate → overlay reuses the cached tracks.parquet (no re-detect).
      const jb = await supabase
        .from('processing_jobs')
        .insert({ match_id: ctx.matchId, video_id: ctx.videoId, stage: 'remap', status: 'queued', params: {} })
      if (jb.error) throw jb.error
      navigate(`/match/${slug}`)
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : 'Speichern fehlgeschlagen')
      setSaving(false)
    }
  }

  if (phase === 'loading') return <Centered>Lädt…</Centered>
  if (phase === 'error' || !ctx) return <Centered>{errorMsg ?? 'Spiel nicht gefunden.'}</Centered>

  return (
    <main style={{ minHeight: '100%', background: 'var(--beige, #FEF4EA)', color: 'var(--ink, #0A0A0B)', fontFamily: 'var(--font-body)' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: isDesktop ? 24 : 14 }}>
        <button
          type="button"
          onClick={() => navigate(`/match/${slug}`)}
          style={{ border: 'none', background: 'none', color: 'var(--muted)', fontSize: 14, cursor: 'pointer', padding: 0, marginBottom: 8 }}
        >
          ‹ Zurück zum Bericht
        </button>
        <h1 style={{ fontFamily: 'var(--font-head)', fontSize: isDesktop ? 28 : 22, margin: '0 0 16px' }}>Spieler zuordnen</h1>

        <PlayerReviewBoard
          initialRows={ctx.rows}
          players={ctx.players}
          pathByLabel={ctx.pathByLabel}
          H={ctx.H}
          frameUrl={ctx.frameUrl}
          size={ctx.size}
          discipline={ctx.discipline}
          saving={saving}
          saveError={saveError}
          onConfirm={onConfirm}
          onSave={onSave}
        />
      </div>
    </main>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
}
