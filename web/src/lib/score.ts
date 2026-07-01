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

/** Viewer-perspective result + line for the report header. The canonical `matches.result_side_a`
 * is authoritative when set (counting `games` alone yields "Sieg 0:0" on missing games and can
 * disagree with the match list, which reads the column); the game count only fills in the numbers.
 * Falls back to counting games when no canonical result is stored; null when neither exists. */
export function matchResult(
  games: number[][],
  resultSideA: MatchResultCode | null,
  viewerSide: Side,
): { result: MatchResultCode | null; line: string } {
  const [a, b] = gamesWon(games)
  const [won, lost] = viewerSide === 'A' ? [a, b] : [b, a]
  const result: MatchResultCode | null = resultSideA
    ? viewerSide === 'A' ? resultSideA : resultSideA === 'win' ? 'loss' : 'win'
    : games.length ? (won >= lost ? 'win' : 'loss') : null
  if (!result) return { result: null, line: '' }
  const counts = games.length ? ` ${won}:${lost}` : ''
  return { result, line: `${result === 'win' ? 'Sieg' : 'Niederlage'}${counts}` }
}
