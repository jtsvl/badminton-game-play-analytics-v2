import { describe, it, expect } from 'vitest'
import { parseScore, gamesWon, matchResult, resultForSideA, resultLine } from './score'

describe('score', () => {
  const games = [[21, 18], [19, 21], [21, 15]]

  it('parses a score string', () => {
    expect(parseScore('21:18 · 19:21 · 21:15')).toEqual(games)
  })
  it('counts games won', () => {
    expect(gamesWon(games)).toEqual([2, 1])
  })
  it('result for side A', () => {
    expect(resultForSideA(games)).toBe('win')
    expect(resultForSideA([[18, 21], [21, 23]])).toBe('loss')
  })
  it('result line from viewer perspective', () => {
    expect(resultLine(games, 'A')).toBe('Sieg 2:1')
    expect(resultLine(games, 'B')).toBe('Niederlage 1:2')
  })
})

describe('matchResult', () => {
  const games = [[21, 18], [19, 21], [21, 15]]

  it('uses the canonical result when games are empty (no "Sieg 0:0")', () => {
    expect(matchResult([], 'win', 'A')).toEqual({ result: 'win', line: 'Sieg' })
    expect(matchResult([], 'loss', 'A')).toEqual({ result: 'loss', line: 'Niederlage' })
  })
  it('flips the canonical result for viewer side B', () => {
    expect(matchResult(games, 'win', 'B')).toEqual({ result: 'loss', line: 'Niederlage 1:2' })
    expect(matchResult([], 'win', 'B')).toEqual({ result: 'loss', line: 'Niederlage' })
  })
  it('falls back to counting games when no canonical result is stored', () => {
    expect(matchResult(games, null, 'A')).toEqual({ result: 'win', line: 'Sieg 2:1' })
    expect(matchResult(games, null, 'B')).toEqual({ result: 'loss', line: 'Niederlage 1:2' })
  })
  it('canonical result wins over the game count when they disagree', () => {
    // games say A won 2:1, but the stored result says A lost — the stored word is authoritative
    expect(matchResult(games, 'loss', 'A')).toEqual({ result: 'loss', line: 'Niederlage 2:1' })
  })
  it('returns null when there is neither a canonical result nor games', () => {
    expect(matchResult([], null, 'A')).toEqual({ result: null, line: '' })
  })
})
