import { describe, it, expect } from 'vitest'
import { deNum, deKm, dateParts, fmtTime } from './format'

describe('format (de-DE)', () => {
  it('formats integers with thousands dot', () => {
    expect(deNum(2840)).toBe('2.840')
  })
  it('formats decimals with comma', () => {
    expect(deNum(5.9, 1)).toBe('5,9')
  })
  it('meters → km with 2 decimals', () => {
    expect(deKm(2840)).toBe('2,84')
  })
  it('derives German date parts', () => {
    const p = dateParts('2026-06-10')
    expect(p.day).toBe('10')
    expect(p.month).toBe('Juni')
    expect(p.dateLabel).toBe('10. Juni 2026')
    expect(p.dd).toBe('10.06.')
  })
  it('formats time as M:SS', () => {
    expect(fmtTime(154)).toBe('2:34')
    expect(fmtTime(5)).toBe('0:05')
  })
})
