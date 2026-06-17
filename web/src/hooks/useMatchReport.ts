import { useEffect, useState } from 'react'
import { supabase, isConfigured } from '../lib/supabase'
import { dateParts } from '../lib/format'
import { gamesWon } from '../lib/score'
import { shotDE } from '../lib/labels'
import { resolveVideoSrc } from '../lib/storage'
import type { MatchReport, ReportPlayer, ShotPair, Side, MatchResultCode } from '../types'

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
        const m = match as any
        const mid = m.id as string

        const [mpRes, pmsRes, pssRes, pairRes, rallyRes, vidRes, endRes] = await Promise.all([
          supabase.from('match_players').select('side, slot, player_id, players(name, first_name)').eq('match_id', mid),
          supabase.from('player_match_stats').select('*').eq('match_id', mid),
          supabase.from('player_shot_stats').select('player_id, shot_type, count').eq('match_id', mid),
          supabase.from('pair_match_stats').select('*').eq('match_id', mid),
          supabase.from('match_rally_stats').select('*').eq('match_id', mid).maybeSingle(),
          supabase.from('match_videos').select('*').eq('match_id', mid).limit(1),
          supabase.from('rallies').select('end_shot_type').eq('match_id', mid),
        ])

        const mps = (mpRes.data ?? []) as any[]
        const statsByPlayer = new Map<string, any>(((pmsRes.data ?? []) as any[]).map((s) => [s.player_id, s]))
        const shotsByPlayer = new Map<string, ShotPair[]>()
        for (const r of (pssRes.data ?? []) as any[]) {
          const list = shotsByPlayer.get(r.player_id) ?? []
          list.push([r.shot_type, r.count])
          shotsByPlayer.set(r.player_id, list)
        }

        const games: number[][] = Array.isArray(m.games) ? m.games : []
        const viewerSide: Side =
          (mps.find((p) => p.player_id === viewerPlayerId)?.side as Side) ??
          (mps.find((p) => p.side === 'A')?.side as Side) ?? 'A'
        const [aWon, bWon] = gamesWon(games)
        const [vWon, vLost] = viewerSide === 'A' ? [aWon, bWon] : [bWon, aWon]
        const result: MatchResultCode = vWon >= vLost ? 'win' : 'loss'
        const resultLine = `${vWon >= vLost ? 'Sieg' : 'Niederlage'} ${vWon}:${vLost}`

        const players: ReportPlayer[] = mps.map((mp) => {
          const st = statsByPlayer.get(mp.player_id)
          const shots = (shotsByPlayer.get(mp.player_id) ?? []).sort((x, y) => y[1] - x[1])
          const player = Array.isArray(mp.players) ? mp.players[0] : mp.players
          return {
            name: player?.name ?? 'Unbekannt',
            first: player?.first_name ?? '',
            isViewer: mp.player_id === viewerPlayerId,
            side: mp.side as Side,
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
            },
          }
        })
        // viewer first
        players.sort((a, b) => Number(b.isViewer) - Number(a.isViewer))

        // endLine from ending-shot distribution
        const endings = ((endRes.data ?? []) as any[]).map((r) => r.end_shot_type).filter(Boolean)
        let endLine = ''
        if (endings.length) {
          const counts = new Map<string, number>()
          for (const e of endings) counts.set(e, (counts.get(e) ?? 0) + 1)
          const [topShot, topCount] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
          endLine = `Der ${shotDE(topShot)} beendete ${Math.round((topCount / endings.length) * 100)} % aller Ballwechsel.`
        }

        const rally = rallyRes.data as any
        const dp = dateParts(m.played_on)
        const vid = ((vidRes.data ?? []) as any[])[0]

        const report: MatchReport = {
          video: vid ? { src: resolveVideoSrc(vid.storage_path), label: vid.label ?? '' } : null,
          meta: {
            id: mid, dateLabel: dp.dateLabel, weekday: m.weekday, dd: dp.dd,
            discipline: m.discipline, durationMin: m.duration_min, restMin: m.rest_min,
            court: m.court, score: m.score, games, result, resultLine,
          },
          rallies: rally
            ? {
                count: rally.count, avgShots: rally.avg_shots, avgSec: rally.avg_sec,
                longShots: rally.long_shots ?? 0, longSec: rally.long_sec ?? 0, endLine,
              }
            : null,
          pairs: ((pairRes.data ?? []) as any[]).map((p) => ({
            side: p.side, label: p.label, fb: p.fb, ss: p.ss, rot: p.rot,
          })),
          players,
        }
        if (!cancelled) setState({ loading: false, error: null, report })
      } catch (e: any) {
        if (!cancelled) setState({ loading: false, error: e?.message ?? 'Fehler beim Laden', report: null })
      }
    })()
    return () => { cancelled = true }
  }, [slug, viewerPlayerId])

  return state
}
