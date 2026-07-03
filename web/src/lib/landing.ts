// „Aus-Schläge" aus player_shot_stats.errors — vom Worker aus Rally-Ende-Landungen geschätzt
// (specs/features/shuttle-landing-inout.md). Eine Untergrenze: nur sicher erkannte Landungen zählen.

/** Summe der errors-Spalten eines Spielers; null, wenn keine Zeile errors trägt (Feature nie gelaufen). */
export function sumErrors(rows: Array<{ errors: number | null }>): number | null {
  const vals = rows.map((r) => r.errors).filter((e): e is number => e != null)
  if (vals.length === 0) return null
  return vals.reduce((a, b) => a + b, 0)
}
