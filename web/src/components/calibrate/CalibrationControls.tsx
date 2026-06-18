interface Props {
  status: string | null
  valid: boolean
  saving: boolean
  onReset: () => void
  onSave: () => void
  error?: string | null
}

const STATUS_DE: Record<string, string> = {
  proposed: 'Vorschlag',
  needs_review: 'Zu prüfen',
  accepted: 'Bestätigt',
  finalized: 'Fertig',
  rejected: 'Verworfen',
}

const btn = (primary: boolean, disabled: boolean): React.CSSProperties => ({
  padding: '10px 16px',
  borderRadius: 10,
  border: primary ? 'none' : '1px solid var(--hairline)',
  background: primary ? (disabled ? '#E8B589' : 'var(--crusta, #FA8832)') : 'var(--card, #fff)',
  color: primary ? '#1A0500' : 'var(--ink)',
  fontWeight: 700,
  fontSize: 14,
  fontFamily: 'var(--font-body)',
  cursor: disabled ? 'not-allowed' : 'pointer',
  opacity: disabled && !primary ? 0.6 : 1,
})

/** Status pill + reset/save + validity/error hint for the calibration screen. */
export function CalibrationControls({ status, valid, saving, onReset, onSave, error }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {status && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--muted)' }}>Status</span>
          <span
            style={{
              fontSize: 12,
              fontWeight: 700,
              padding: '3px 10px',
              borderRadius: 999,
              background: '#FEF4EA',
              color: '#1A0500',
              border: '1px solid var(--hairline)',
            }}
          >
            {STATUS_DE[status] ?? status}
          </span>
        </div>
      )}

      <p style={{ margin: 0, fontSize: 13, color: valid ? 'var(--muted)' : '#C0392B', lineHeight: 1.5 }}>
        {valid
          ? 'Zieh die vier Ecken genau auf die Eckpunkte des Spielfelds, bis das orange Gitter auf den weißen Linien liegt. Beim Ziehen zeigt eine Lupe die genaue Position.'
          : 'Die vier Ecken müssen ein gültiges Viereck bilden (keine Überkreuzung).'}
      </p>

      {error && <p style={{ margin: 0, fontSize: 13, color: '#C0392B' }}>{error}</p>}

      <div style={{ display: 'flex', gap: 10 }}>
        <button type="button" style={btn(false, saving)} disabled={saving} onClick={onReset}>
          Zurücksetzen
        </button>
        <button type="button" style={btn(true, !valid || saving)} disabled={!valid || saving} onClick={onSave}>
          {saving ? 'Speichert…' : 'Kalibrierung speichern'}
        </button>
      </div>
    </div>
  )
}
