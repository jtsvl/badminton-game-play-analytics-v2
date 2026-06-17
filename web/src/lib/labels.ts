// German display labels for canonical stored values. DB stores neutral/English; the UI localizes here.
import type { Discipline, MatchResultCode } from '../types'

export const disciplineDE = (d: Discipline): string => (d === 'doubles' ? 'Doppel' : 'Einzel')

export const resultDE = (r: MatchResultCode | null): string =>
  r === 'win' ? 'Sieg' : r === 'loss' ? 'Niederlage' : ''

// canonical shot_type → German label shown in the report (FR-21/22)
export const SHOT_DE: Record<string, string> = {
  clear: 'Clear',
  smash: 'Smash',
  drop: 'Drop',
  net: 'Netzdrop',
  drive: 'Drive',
  lift: 'Lift',
  serve: 'Aufschlag',
  unknown: 'Sonstige',
}
export const shotDE = (s: string): string => SHOT_DE[s] ?? s
