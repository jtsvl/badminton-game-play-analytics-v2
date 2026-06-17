// German (de-DE) formatting helpers — decimal comma, thousands dot. See specs/50-web-app.md.

const MONTHS_DE = [
  'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
  'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember',
]

export function deNum(value: number, decimals = 0): string {
  return value.toLocaleString('de-DE', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/** Meters → km string with 2 decimals (e.g. 2840 → "2,84"). */
export function deKm(meters: number): string {
  return deNum(meters / 1000, 2)
}

export interface DateParts { day: string; month: string; dateLabel: string; dd: string }

/** ISO date (yyyy-mm-dd) → German display parts. */
export function dateParts(iso: string): DateParts {
  const d = new Date(`${iso}T00:00:00`)
  const day = d.getDate()
  const mm = d.getMonth()
  return {
    day: String(day),
    month: MONTHS_DE[mm],
    dateLabel: `${day}. ${MONTHS_DE[mm]} ${d.getFullYear()}`,
    dd: `${String(day).padStart(2, '0')}.${String(mm + 1).padStart(2, '0')}.`,
  }
}

/** Seconds → "M:SS". */
export function fmtTime(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60)
  const s = Math.floor(totalSeconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
