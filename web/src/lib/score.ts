// Score parsing / result helpers. See specs/50-web-app.md.
import type { MatchResultCode, Side } from '../types'

/** "21:18 · 19:21 · 21:15" → [[21,18],[19,21],[21,15]] */
export function parseScore(score: string | null): number[][] {
  if (!score) return []
  return score
    .split('·')
    .map((g) => g.trim())
    .filter(Boolean)
    .map((g) => g.split(':').map((n) => parseInt(n.trim(), 10)))
}

/** Games won by [sideA, sideB]. */
export function gamesWon(games: number[][]): [number, number] {
  let a = 0
  let b = 0
  for (const [x, y] of games) {
    if (x > y) a++
    else if (y > x) b++
  }
  return [a, b]
}

export function resultForSideA(games: number[][]): MatchResultCode {
  const [a, b] = gamesWon(games)
  return a >= b ? 'win' : 'loss'
}

/** From the viewer's perspective: "Sieg 2:1" / "Niederlage 0:2". */
export function resultLine(games: number[][], viewerSide: Side): string {
  const [a, b] = gamesWon(games)
  const [won, lost] = viewerSide === 'A' ? [a, b] : [b, a]
  return `${won >= lost ? 'Sieg' : 'Niederlage'} ${won}:${lost}`
}
