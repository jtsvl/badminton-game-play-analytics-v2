// TS types derived from specs/20-data-model.md. The DB rows are the worker↔web contract;
// the assembled `MatchReport` mirrors handovers/**/mock-data.js shapes.

import type { Pt } from './lib/homography'
import type { TrackKey } from './lib/overlay'

export type Discipline = 'singles' | 'doubles'
export type Side = 'A' | 'B'
export type MatchResultCode = 'win' | 'loss'

// ── DB rows ──
export interface PlayerRow {
  id: string; name: string; first_name: string; email: string | null; is_member: boolean
}
export interface MatchRow {
  id: string; slug: string; played_on: string; weekday: string | null; discipline: Discipline
  court: number | null; score: string | null; games: number[][]
  duration_min: number | null; rest_min: number | null
  result_side_a: MatchResultCode | null; status: string
}
export interface MatchVideoRow {
  id: string; match_id: string; storage_path: string; label: string | null
  angle: 'overhead' | 'side'; duration_s: number | null; fps: number | null
  width: number | null; height: number | null
}
// Calibration is its own table now (court_calibrations) — see specs/20-data-model.md (P1).
export interface MatchPlayerRow {
  id: string; match_id: string; player_id: string; side: Side; slot: number
}
export interface Zones { front: number; mid: number; rear: number; left: number; right: number }
export interface HeatKernel { x: number; y: number; w: number; s: number }
export interface PlayerMatchStatsRow {
  match_id: string; player_id: string
  distance_m: number; avg_kmh: number; top_kmh: number; sprints: number
  zones: Zones; timeline: number[]; heat: HeatKernel[]; heatmap_uri: string | null
  fastest: number | null
}
export interface PlayerShotStatRow { match_id: string; player_id: string; shot_type: string; count: number }
export interface PairMatchStatRow { match_id: string; side: Side; label: string; fb: number; ss: number; rot: number }
export interface MatchRallyStatsRow {
  match_id: string; count: number; avg_shots: number; avg_sec: number
  long_shots: number | null; long_sec: number | null
}

// ── assembled report (frontend shape) ──
export type ShotPair = [string, number]
export interface RallySummary { count: number; avgShots: number; avgSec: number; longShots: number; longSec: number; endLine: string }
export interface PairStat { side: Side; label: string; fb: number; ss: number; rot: number }
export interface ReportPlayer {
  playerId: string // needed to map a player to its track_keyframes subject ('player:<playerId>')
  name: string; first: string; isViewer: boolean; side: Side
  stats: Omit<PlayerMatchStatsRow, 'match_id' | 'player_id'> & { shots: ShotPair[] }
}
export interface ReportMeta {
  id: string; dateLabel: string; weekday: string | null; dd: string; discipline: Discipline
  durationMin: number | null; restMin: number | null; court: number | null
  score: string | null; games: number[][]; result: MatchResultCode | null; resultLine: string
}
// Court calibration consumed by the overlay (the 4 confirmed corners are the source of truth;
// the frontend derives court→frame via lib/calibration.homographyFromCorners).
export interface CourtCalib {
  confirmedPoints: Pt[] | null
  imageSize: { w: number; h: number } | null
}
// One track_keyframes row: subject 'player:<id>' | 'shuttle', data [{t,u,v}] (t in seconds @ ~10 Hz).
export interface TrackData {
  subject: string
  data: TrackKey[]
}
export interface MatchReport {
  video: { src: string; label: string; durationS: number | null } | null
  calib: CourtCalib | null
  tracks: TrackData[]
  meta: ReportMeta
  rallies: RallySummary | null
  pairs: PairStat[] | null
  players: ReportPlayer[]
}

// ── match list item ──
export interface MatchListItem {
  id: string; slug: string; dd: string; day: string; month: string; dateLabel: string
  discipline: Discipline; a: string; b: string; score: string | null
  result: MatchResultCode | null; isNew: boolean; needsCalibration: boolean
}
