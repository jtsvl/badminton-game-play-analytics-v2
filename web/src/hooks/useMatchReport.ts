import { useEffect, useState } from 'react'
import { supabase, isConfigured } from '../lib/supabase'
import { dateParts } from '../lib/format'
import { sumErrors } from '../lib/landing'
import { matchResult } from '../lib/score'
import { shotDE } from '../lib/labels'
import { resolvePlaybackSrc } from '../lib/storage'
import type {
  CourtCalibrationPointsRow, MatchPlayerJoinRow, MatchRallyStatsRow, MatchReport, MatchRow,
  MatchVideoRow, PairMatchStatRow, PlayerMatchStatsRow, PlayerShotStatRow, RallyEndRow,
  ReportPlayer, ShotPair, Side, TrackKeyframeRow,
} from '../types'

interface ReportState { loading: boolean; error: string | null; report: MatchReport | null }

/** Loads a match by slug and assembles the frontend report object (specs/20-data-model.md mapping). */
export function useMatchReport(slug: string | undefined, viewerPlayerId?: string): ReportState {
  const [state, setState] = useState<ReportState>({ loading: true, error: null, report: null })

  useEffect(() => {
    if (!slug) return
    if (!isConfigured) {
      setState({ loading: false, error: 'Supabase ist nicht konfiguriert (.env).', report: null })
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const { data: match, error: matchErr } = await supabase
          .from('matches').select('*').eq('slug', slug).single()
        if (matchErr || !match) throw matchErr ?? new Error('Spiel nicht gefunden')
        const m = match as MatchRow
        const mid = m.id

        const [mpRes, pmsRes, pssRes, pairRes, rallyRes, vidRes, endRes] = await Promise.all([
          supabase.from('match_players').select('side, slot, player_id, players(name, first_name)').eq('match_id', mid),
          supabase.from('player_match_stats').select('*').eq('match_id', mid),
          supabase.from('player_shot_stats').select('player_id, shot_type, count, errors').eq('match_id', mid),
          supabase.from('pair_match_stats').select('*').eq('match_id', mid),
          supabase.from('match_rally_stats').select('*').eq('match_id', mid).maybeSingle(),
          supabase.from('match_videos').select('*').eq('match_id', mid).limit(1),
          supabase.from('rallies').select('end_shot_type').eq('match_id', mid),
        ])

        const mps = (mpRes.data ?? []) as MatchPlayerJoinRow[]
        const statsByPlayer = new Map<string, PlayerMatchStatsRow>(
          ((pmsRes.data ?? []) as PlayerMatchStatsRow[]).map((s) => [s.player_id, s]))
        const shotsByPlayer = new Map<string, ShotPair[]>()
        const errorRowsByPlayer = new Map<string, PlayerShotStatRow[]>()
        for (const r of (pssRes.data ?? []) as PlayerShotStatRow[]) {
          // errors-only rows (unknown/count=0, vom Landing-Feature) dürfen keinen 0-Balken rendern
          if (r.count > 0) {
            const list = shotsByPlayer.get(r.player_id) ?? []
            list.push([r.shot_type, r.count])
            shotsByPlayer.set(r.player_id, list)
          }
          const errs = errorRowsByPlayer.get(r.player_id) ?? []
          errs.push(r)
          errorRowsByPlayer.set(r.player_id, errs)
        }

        const games: number[][] = Array.isArray(m.games) ? m.games : []
        const viewerSide: Side =
          mps.find((p) => p.player_id === viewerPlayerId)?.side ??
          mps.find((p) => p.side === 'A')?.side ?? 'A'
        const { result, line: resultLine } = matchResult(games, m.result_side_a ?? null, viewerSide)

        const players: ReportPlayer[] = mps.map((mp) => {
          const st = statsByPlayer.get(mp.player_id)
          const shots = (shotsByPlayer.get(mp.player_id) ?? []).sort((x, y) => y[1] - x[1])
          const player = Array.isArray(mp.players) ? mp.players[0] : mp.players
          return {
            playerId: mp.player_id,
            name: player?.name ?? 'Unbekannt',
            first: player?.first_name ?? '',
            isViewer: mp.player_id === viewerPlayerId,
            side: mp.side,
            stats: {
              distance_m: st?.distance_m ?? 0,
              avg_kmh: st?.avg_kmh ?? 0,
              top_kmh: st?.top_kmh ?? 0,
              sprints: st?.sprints ?? 0,
              zones: st?.zones ?? { front: 0, mid: 0, rear: 0, left: 0, right: 0 },
              timeline: st?.timeline ?? [],
              heat: st?.heat ?? [],
              heatmap_uri: st?.heatmap_uri ?? null,
              fastest: st?.fastest ?? null,
              shots,
              outShots: sumErrors(errorRowsByPlayer.get(mp.player_id) ?? []),
            },
          }
        })
        // viewer first
        players.sort((a, b) => Number(b.isViewer) - Number(a.isViewer))

        // endLine from ending-shot distribution
        const endings = ((endRes.data ?? []) as RallyEndRow[])
          .map((r) => r.end_shot_type)
          .filter((t): t is string => Boolean(t))
        let endLine = ''
        if (endings.length) {
          const counts = new Map<string, number>()
          for (const e of endings) counts.set(e, (counts.get(e) ?? 0) + 1)
          const [topShot, topCount] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
          endLine = `Der ${shotDE(topShot)} beendete ${Math.round((topCount / endings.length) * 100)} % aller Ballwechsel.`
        }

        const rally = rallyRes.data as MatchRallyStatsRow | null
        const dp = dateParts(m.played_on)
        const vid = ((vidRes.data ?? []) as MatchVideoRow[])[0]

        // Calibration + tracks key on vid.id (known only after the first batch) — one dependent fetch.
        let calib: MatchReport['calib'] = null
        let tracks: MatchReport['tracks'] = []
        let videoSrc = ''
        if (vid) {
          const [srcUrl, calibRes, trackRes] = await Promise.all([
            resolvePlaybackSrc(vid),
            supabase
              .from('court_calibrations')
              .select('confirmed_points, image_size')
              .eq('video_id', vid.id)
              .order('updated_at', { ascending: false })
              .limit(1)
              .maybeSingle(),
            supabase.from('track_keyframes').select('subject, data').eq('video_id', vid.id),
          ])
          videoSrc = srcUrl
          const cc = calibRes.data as CourtCalibrationPointsRow | null
          calib = cc
            ? { confirmedPoints: cc.confirmed_points ?? null, imageSize: cc.image_size ?? null }
            : null
          tracks = ((trackRes.data ?? []) as TrackKeyframeRow[])
            .map((t) => ({ subject: t.subject, data: t.data }))
        }

        const report: MatchReport = {
          video: vid
            ? { id: vid.id, src: videoSrc, label: vid.label ?? '', durationS: vid.duration_s ?? null }
            : null,
          calib,
          tracks,
          meta: {
            id: mid, dateLabel: dp.dateLabel, weekday: m.weekday, dd: dp.dd,
            discipline: m.discipline, durationMin: m.duration_min, restMin: m.rest_min,
            court: m.court, score: m.score, games, result, resultLine,
            status: m.status,
          },
          rallies: rally
            ? {
                count: rally.count, avgShots: rally.avg_shots, avgSec: rally.avg_sec,
                longShots: rally.long_shots ?? 0, longSec: rally.long_sec ?? 0, endLine,
              }
            : null,
          pairs: ((pairRes.data ?? []) as PairMatchStatRow[]).map((p) => ({
            side: p.side, label: p.label, fb: p.fb, ss: p.ss, rot: p.rot,
          })),
          players,
        }
        if (!cancelled) setState({ loading: false, error: null, report })
      } catch (e) {
        // Supabase errors are plain objects with .message, not Error instances — narrow both.
        const message = e instanceof Error
          ? e.message
          : (e as { message?: string } | null | undefined)?.message
        if (!cancelled) setState({ loading: false, error: message ?? 'Fehler beim Laden', report: null })
      }
    })()
    return () => { cancelled = true }
  }, [slug, viewerPlayerId])

  return state
}
