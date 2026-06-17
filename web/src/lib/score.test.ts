import { describe, it, expect } from 'vitest'
import { parseScore, gamesWon, resultForSideA, resultLine } from './score'

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
