import { describe, expect, it } from 'vitest'
import { sumErrors } from './landing'

describe('sumErrors', () => {
  it('summiert errors über alle shot_type-Zeilen eines Spielers', () => {
    expect(sumErrors([{ errors: 2 }, { errors: 1 }])).toBe(3)
  })

  it('null-errors-Zeilen zählen nicht mit', () => {
    expect(sumErrors([{ errors: 2 }, { errors: null }])).toBe(2)
  })

  it('0 ist ein echter Wert (Feature lief, keine Aus-Schläge) — nicht verstecken', () => {
    expect(sumErrors([{ errors: 0 }])).toBe(0)
  })

  it('null, wenn keine Zeile errors trägt (Feature nie gelaufen → Zeile ausblenden)', () => {
    expect(sumErrors([{ errors: null }, { errors: null }])).toBeNull()
    expect(sumErrors([])).toBeNull()
  })
})
