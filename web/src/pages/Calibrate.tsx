import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { supabase, isConfigured } from '../lib/supabase'
import { signedUrl } from '../lib/storage'
import { homographyFromCorners, isConvexQuad } from '../lib/calibration'
import type { Pt } from '../lib/homography'
import { useMediaQuery } from '../hooks/useMediaQuery'
import { CalibrationCanvas } from '../components/calibrate/CalibrationCanvas'
import { CalibrationControls } from '../components/calibrate/CalibrationControls'

interface Ctx {
  matchId: string
  videoId: string
  calibId: string
  size: { w: number; h: number }
  frameUrl: string | null
  status: string | null
  proposed: Pt[]
}

function defaultCorners(size: { w: number; h: number }): Pt[] {
  const { w, h } = size
  return [
    [0.3 * w, 0.32 * h], // FL
    [0.7 * w, 0.32 * h], // FR
    [0.88 * w, 0.92 * h], // NR
    [0.12 * w, 0.92 * h], // NL
  ]
}

function safeH(corners: Pt[], size: { w: number; h: number }): number[] | null {
  try {
    return homographyFromCorners(corners, size)
  } catch {
    return null
  }
}

/** /match/:slug/calibrate — drag the four court corners; Save persists them and starts processing. */
export function Calibrate() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const isDesktop = useMediaQuery('(min-width: 1200px)')
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [ctx, setCtx] = useState<Ctx | null>(null)
  const [corners, setCorners] = useState<Pt[] | null>(null)
  const [lastH, setLastH] = useState<number[] | null>(null)
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
        const { data: match } = await supabase.from('matches').select('id, status').eq('slug', slug).single()
        if (!match) throw new Error('Spiel nicht gefunden')
        const { data: video } = await supabase
          .from('match_videos')
          .select('id, storage_path, width, height')
          .eq('match_id', match.id)
          .limit(1)
          .maybeSingle()
        if (!video) throw new Error('Kein Video für dieses Spiel')
        const { data: calib } = await supabase
          .from('court_calibrations')
          .select('id, status, frame_uri, image_size, proposed_points, confirmed_points')
          .eq('video_id', video.id)
          .order('updated_at', { ascending: false })
          .limit(1)
          .maybeSingle()
        if (!calib) throw new Error('Noch keine Kalibrierung vorbereitet (Ingest ausstehend).')

        const size = (calib.image_size as { w: number; h: number } | null) ?? {
          w: video.width ?? 1920,
          h: video.height ?? 1080,
        }
        const frameUrl =
          (calib.frame_uri ? await signedUrl('analysis-artifacts', calib.frame_uri) : '') ||
          (await signedUrl('match-videos', video.storage_path)) ||
          null
        const proposed = (calib.proposed_points as Pt[] | null) ?? defaultCorners(size)
        const start = (calib.confirmed_points as Pt[] | null) ?? proposed

        if (cancelled) return
        setCtx({
          matchId: match.id,
          videoId: video.id,
          calibId: calib.id,
          size,
          frameUrl,
          status: calib.status ?? null,
          proposed,
        })
        setCorners(start)
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

  const valid = corners ? isConvexQuad(corners) : false
  const H = useMemo(() => {
    if (!corners || !ctx) return lastH
    const h = valid ? safeH(corners, ctx.size) : null
    return h ?? lastH
  }, [corners, ctx, valid, lastH])
  useEffect(() => {
    if (valid && corners && ctx) {
      const h = safeH(corners, ctx.size)
      if (h) setLastH(h)
    }
  }, [valid, corners, ctx])

  const onDrag = (i: number, px: Pt) =>
    setCorners((cur) => (cur ? cur.map((c, j) => (j === i ? px : c)) : cur))

  const onReset = () => ctx && setCorners(ctx.proposed)

  const onSave = async () => {
    if (!ctx || !corners) return
    setSaving(true)
    setSaveError(null)
    try {
      const up = await supabase
        .from('court_calibrations')
        .update({ confirmed_points: corners, status: 'accepted', source: 'manual', updated_at: new Date().toISOString() })
        .eq('id', ctx.calibId)
      if (up.error) throw up.error
      const ms = await supabase.from('matches').update({ status: 'processing' }).eq('id', ctx.matchId)
      if (ms.error) throw ms.error
      const jb = await supabase
        .from('processing_jobs')
        .insert({ match_id: ctx.matchId, video_id: ctx.videoId, stage: 'calibrate', status: 'queued', params: { calibration_id: ctx.calibId } })
      if (jb.error) throw jb.error
      navigate(`/match/${slug}`)
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : 'Speichern fehlgeschlagen')
      setSaving(false)
    }
  }

  if (phase === 'loading') return <Centered>Lädt…</Centered>
  if (phase === 'error' || !ctx || !corners || !H) return <Centered>{errorMsg ?? 'Kein Standbild gefunden.'}</Centered>

  const controls = (
    <CalibrationControls status={ctx.status} valid={valid} saving={saving} onReset={onReset} onSave={onSave} error={saveError} />
  )

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
        <h1 style={{ fontFamily: 'var(--font-head)', fontSize: isDesktop ? 28 : 22, margin: '0 0 16px' }}>Feld kalibrieren</h1>

        <div style={{ display: isDesktop ? 'grid' : 'block', gridTemplateColumns: isDesktop ? 'minmax(0, 1fr) 300px' : undefined, gap: 20, alignItems: 'start' }}>
          <CalibrationCanvas imgSrc={ctx.frameUrl} size={ctx.size} corners={corners} H={H} valid={valid} onDrag={onDrag} />
          <div style={{ marginTop: isDesktop ? 0 : 16 }}>{controls}</div>
        </div>
      </div>
    </main>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, color: 'var(--muted)', fontSize: 14 }}>{children}</main>
  )
}
